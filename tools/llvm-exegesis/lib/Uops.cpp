//===-- Uops.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Uops.h"

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "PerfHelper.h"
#include "Target.h"

// FIXME: Load constants into registers (e.g. with fld1) to not break
// instructions like x87.

// Ideally we would like the only limitation on executing uops to be the issue
// ports. Maximizing port pressure increases the likelihood that the load is
// distributed evenly across possible ports.

// To achieve that, one approach is to generate instructions that do not have
// data dependencies between them.
//
// For some instructions, this is trivial:
//    mov rax, qword ptr [rsi]
//    mov rax, qword ptr [rsi]
//    mov rax, qword ptr [rsi]
//    mov rax, qword ptr [rsi]
// For the above snippet, haswell just renames rax four times and executes the
// four instructions two at a time on P23 and P0126.
//
// For some instructions, we just need to make sure that the source is
// different from the destination. For example, IDIV8r reads from GPR and
// writes to AX. We just need to ensure that the Var is assigned a
// register which is different from AX:
//    idiv bx
//    idiv bx
//    idiv bx
//    idiv bx
// The above snippet will be able to fully saturate the ports, while the same
// with ax would issue one uop every `latency(IDIV8r)` cycles.
//
// Some instructions make this harder because they both read and write from
// the same register:
//    inc rax
//    inc rax
//    inc rax
//    inc rax
// This has a data dependency from each instruction to the next, limit the
// number of instructions that can be issued in parallel.
// It turns out that this is not a big issue on recent Intel CPUs because they
// have heuristics to balance port pressure. In the snippet above, subsequent
// instructions will end up evenly distributed on {P0,P1,P5,P6}, but some CPUs
// might end up executing them all on P0 (just because they can), or try
// avoiding P5 because it's usually under high pressure from vector
// instructions.
// This issue is even more important for high-latency instructions because
// they increase the idle time of the CPU, e.g. :
//    imul rax, rbx
//    imul rax, rbx
//    imul rax, rbx
//    imul rax, rbx
//
// To avoid that, we do the renaming statically by generating as many
// independent exclusive assignments as possible (until all possible registers
// are exhausted) e.g.:
//    imul rax, rbx
//    imul rcx, rbx
//    imul rdx, rbx
//    imul r8,  rbx
//
// Some instruction even make the above static renaming impossible because
// they implicitly read and write from the same operand, e.g. ADC16rr reads
// and writes from EFLAGS.
// In that case we just use a greedy register assignment and hope for the
// best.

namespace exegesis {

static bool hasUnknownOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_UNKNOWN;
}

llvm::Error
UopsSnippetGenerator::isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const {
  if (llvm::any_of(MCInstrDesc.operands(), hasUnknownOperand))
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has unknown operands");
  return llvm::Error::success();
}

// Returns whether this Variable ties Use and Def operands together.
static bool hasTiedOperands(const Instruction &Instr, const Variable &Var) {
  bool HasUse = false;
  bool HasDef = false;
  for (const unsigned OpIndex : Var.TiedOperands) {
    const Operand &Op = Instr.Operands[OpIndex];
    if (Op.IsDef)
      HasDef = true;
    else
      HasUse = true;
  }
  return HasUse && HasDef;
}

static llvm::SmallVector<const Variable *, 8>
getTiedVariables(const Instruction &Instr) {
  llvm::SmallVector<const Variable *, 8> Result;
  for (const auto &Var : Instr.Variables)
    if (hasTiedOperands(Instr, Var))
      Result.push_back(&Var);
  return Result;
}

static void remove(llvm::BitVector &a, const llvm::BitVector &b) {
  assert(a.size() == b.size());
  for (auto I : b.set_bits())
    a.reset(I);
}

UopsBenchmarkRunner::~UopsBenchmarkRunner() = default;
UopsSnippetGenerator::~UopsSnippetGenerator() = default;

void UopsSnippetGenerator::instantiateMemoryOperands(
    const unsigned ScratchSpacePointerInReg,
    std::vector<InstructionBuilder> &Instructions) const {
  if (ScratchSpacePointerInReg == 0)
    return; // no memory operands.
  const auto &ET = State.getExegesisTarget();
  const unsigned MemStep = ET.getMaxMemoryAccessSize();
  const size_t OriginalInstructionsSize = Instructions.size();
  size_t I = 0;
  for (InstructionBuilder &IB : Instructions) {
    ET.fillMemoryOperands(IB, ScratchSpacePointerInReg, I * MemStep);
    ++I;
  }

  while (Instructions.size() < kMinNumDifferentAddresses) {
    InstructionBuilder IB = Instructions[I % OriginalInstructionsSize];
    ET.fillMemoryOperands(IB, ScratchSpacePointerInReg, I * MemStep);
    ++I;
    Instructions.push_back(std::move(IB));
  }
  assert(I * MemStep < BenchmarkRunner::ScratchSpace::kSize &&
         "not enough scratch space");
}

llvm::Expected<CodeTemplate>
UopsSnippetGenerator::generateCodeTemplate(unsigned Opcode) const {
  const auto &InstrDesc = State.getInstrInfo().get(Opcode);
  if (auto E = isInfeasible(InstrDesc))
    return std::move(E);
  const Instruction Instr(InstrDesc, RATC);
  const auto &ET = State.getExegesisTarget();
  CodeTemplate CT;

  const llvm::BitVector *ScratchSpaceAliasedRegs = nullptr;
  if (Instr.hasMemoryOperands()) {
    CT.ScratchSpacePointerInReg =
        ET.getScratchMemoryRegister(State.getTargetMachine().getTargetTriple());
    if (CT.ScratchSpacePointerInReg == 0)
      return llvm::make_error<BenchmarkFailure>(
          "Infeasible : target does not support memory instructions");
    ScratchSpaceAliasedRegs =
        &RATC.getRegister(CT.ScratchSpacePointerInReg).aliasedBits();
    // If the instruction implicitly writes to ScratchSpacePointerInReg , abort.
    // FIXME: We could make a copy of the scratch register.
    for (const auto &Op : Instr.Operands) {
      if (Op.IsDef && Op.ImplicitReg &&
          ScratchSpaceAliasedRegs->test(*Op.ImplicitReg))
        return llvm::make_error<BenchmarkFailure>(
            "Infeasible : memory instruction uses scratch memory register");
    }
  }

  const AliasingConfigurations SelfAliasing(Instr, Instr);
  InstructionBuilder IB(Instr);
  if (SelfAliasing.empty()) {
    CT.Info = "instruction is parallel, repeating a random one.";
    CT.Instructions.push_back(std::move(IB));
    instantiateMemoryOperands(CT.ScratchSpacePointerInReg, CT.Instructions);
    return std::move(CT);
  }
  if (SelfAliasing.hasImplicitAliasing()) {
    CT.Info = "instruction is serial, repeating a random one.";
    CT.Instructions.push_back(std::move(IB));
    instantiateMemoryOperands(CT.ScratchSpacePointerInReg, CT.Instructions);
    return std::move(CT);
  }
  const auto TiedVariables = getTiedVariables(Instr);
  if (!TiedVariables.empty()) {
    if (TiedVariables.size() > 1)
      return llvm::make_error<llvm::StringError>(
          "Infeasible : don't know how to handle several tied variables",
          llvm::inconvertibleErrorCode());
    const Variable *Var = TiedVariables.front();
    assert(Var);
    assert(!Var->TiedOperands.empty());
    const Operand &Op = Instr.Operands[Var->TiedOperands.front()];
    assert(Op.Tracker);
    CT.Info = "instruction has tied variables using static renaming.";
    for (const llvm::MCPhysReg Reg : Op.Tracker->sourceBits().set_bits()) {
      if (ScratchSpaceAliasedRegs && ScratchSpaceAliasedRegs->test(Reg))
        continue; // Do not use the scratch memory address register.
      InstructionBuilder TmpIB = IB;
      TmpIB.getValueFor(*Var) = llvm::MCOperand::createReg(Reg);
      CT.Instructions.push_back(std::move(TmpIB));
    }
    instantiateMemoryOperands(CT.ScratchSpacePointerInReg, CT.Instructions);
    return std::move(CT);
  }
  // No tied variables, we pick random values for defs.
  llvm::BitVector Defs(State.getRegInfo().getNumRegs());
  for (const auto &Op : Instr.Operands) {
    if (Op.Tracker && Op.IsExplicit && Op.IsDef && !Op.IsMem) {
      auto PossibleRegisters = Op.Tracker->sourceBits();
      remove(PossibleRegisters, RATC.reservedRegisters());
      // Do not use the scratch memory address register.
      if (ScratchSpaceAliasedRegs)
        remove(PossibleRegisters, *ScratchSpaceAliasedRegs);
      assert(PossibleRegisters.any() && "No register left to choose from");
      const auto RandomReg = randomBit(PossibleRegisters);
      Defs.set(RandomReg);
      IB.getValueFor(Op) = llvm::MCOperand::createReg(RandomReg);
    }
  }
  // And pick random use values that are not reserved and don't alias with defs.
  const auto DefAliases = getAliasedBits(State.getRegInfo(), Defs);
  for (const auto &Op : Instr.Operands) {
    if (Op.Tracker && Op.IsExplicit && !Op.IsDef && !Op.IsMem) {
      auto PossibleRegisters = Op.Tracker->sourceBits();
      remove(PossibleRegisters, RATC.reservedRegisters());
      // Do not use the scratch memory address register.
      if (ScratchSpaceAliasedRegs)
        remove(PossibleRegisters, *ScratchSpaceAliasedRegs);
      remove(PossibleRegisters, DefAliases);
      assert(PossibleRegisters.any() && "No register left to choose from");
      const auto RandomReg = randomBit(PossibleRegisters);
      IB.getValueFor(Op) = llvm::MCOperand::createReg(RandomReg);
    }
  }
  CT.Info =
      "instruction has no tied variables picking Uses different from defs";
  CT.Instructions.push_back(std::move(IB));
  instantiateMemoryOperands(CT.ScratchSpacePointerInReg, CT.Instructions);
  return std::move(CT);
}

std::vector<BenchmarkMeasure>
UopsBenchmarkRunner::runMeasurements(const ExecutableFunction &Function,
                                     ScratchSpace &Scratch) const {
  const auto &SchedModel = State.getSubtargetInfo().getSchedModel();

  const auto RunMeasurement = [&Function,
                               &Scratch](const char *const Counters) {
    // We sum counts when there are several counters for a single ProcRes
    // (e.g. P23 on SandyBridge).
    int64_t CounterValue = 0;
    llvm::SmallVector<llvm::StringRef, 2> CounterNames;
    llvm::StringRef(Counters).split(CounterNames, ',');
    for (const auto &CounterName : CounterNames) {
      pfm::PerfEvent UopPerfEvent(CounterName);
      if (!UopPerfEvent.valid())
        llvm::report_fatal_error(
            llvm::Twine("invalid perf event ").concat(Counters));
      pfm::Counter Counter(UopPerfEvent);
      Scratch.clear();
      Counter.start();
      Function(Scratch.ptr());
      Counter.stop();
      CounterValue += Counter.read();
    }
    return CounterValue;
  };

  std::vector<BenchmarkMeasure> Result;
  const auto& PfmCounters = SchedModel.getExtraProcessorInfo().PfmCounters;
  // Uops per port.
  for (unsigned ProcResIdx = 1;
       ProcResIdx < SchedModel.getNumProcResourceKinds(); ++ProcResIdx) {
    const char *const Counters = PfmCounters.IssueCounters[ProcResIdx];
    if (!Counters)
      continue;
    const double CounterValue = RunMeasurement(Counters);
    Result.push_back(BenchmarkMeasure::Create(SchedModel.getProcResource(ProcResIdx)->Name, CounterValue));
  }
  // NumMicroOps.
  if (const char *const UopsCounter = PfmCounters.UopsCounter) {
    const double CounterValue = RunMeasurement(UopsCounter);
    Result.push_back(BenchmarkMeasure::Create("NumMicroOps", CounterValue));
  }
  return Result;
}

constexpr const size_t UopsSnippetGenerator::kMinNumDifferentAddresses;

} // namespace exegesis
