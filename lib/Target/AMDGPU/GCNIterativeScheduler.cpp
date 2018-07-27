//===- GCNIterativeScheduler.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GCNIterativeScheduler.h"
#include "AMDGPUSubtarget.h"
#include "GCNRegPressure.h"
#include "GCNSchedStrategy.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDFS.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Filesystem.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

bool SchedDFSResult2::isParentTree(unsigned PotentailParentSubTreeID,
                                   unsigned PotentialChildSubTreeID) const {
  auto ID = PotentialChildSubTreeID;
  assert(ID != SchedDFSResult::InvalidSubtreeID);
  unsigned ParentID;
  while ((ParentID = DFSTreeData[ID].ParentTreeID) != SchedDFSResult::InvalidSubtreeID) {
    if (ParentID == PotentailParentSubTreeID)
      return true;
    ID = ParentID;
  }
  return false;
}

unsigned SchedDFSResult2::getTopMostParentSubTreeID(const SUnit *Node) const {
  return getTopMostParentSubTreeID(getSubtreeID(Node));
}

unsigned SchedDFSResult2::getTopMostParentSubTreeID(unsigned SubTreeID) const {
  auto ID = SubTreeID;
  assert(ID != SchedDFSResult::InvalidSubtreeID);
  unsigned ParentID;
  while ((ParentID = DFSTreeData[ID].ParentTreeID) != SchedDFSResult::InvalidSubtreeID) {
    ID = ParentID;
  }
  return ID;
}

bool SchedDFSResult2::isInTreeOrDescendant(const SUnit *Node,
                                           unsigned SubTreeID) const {
  auto ID = getSubtreeID(Node);
  if (ID == SubTreeID)
    return true;
  return isParentTree(SubTreeID, ID);
}

namespace llvm {

std::vector<const SUnit *> makeMinRegSchedule(ArrayRef<const SUnit*> TopRoots,
                                              const ScheduleDAG &DAG);

std::vector<const SUnit *> makeMinRegSchedule2(ArrayRef<const SUnit*> BotRoots,
                                               const GCNRPTracker::LiveRegSet &LiveThrRegs,
                                               const GCNRPTracker::LiveRegSet &LiveOutRegs,
                                               const ScheduleDAGMI &DAG);

  std::vector<const SUnit*> makeGCNILPScheduler(ArrayRef<const SUnit*> BotRoots,
    const ScheduleDAG &DAG);
}

// shim accessors for different order containers
static inline MachineInstr *getMachineInstr(MachineInstr *MI) {
  return MI;
}
static inline MachineInstr *getMachineInstr(const SUnit *SU) {
  return SU->getInstr();
}
static inline MachineInstr *getMachineInstr(const SUnit &SU) {
  return SU.getInstr();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
static void printRegion(raw_ostream &OS,
                        MachineBasicBlock::iterator Begin,
                        MachineBasicBlock::iterator End,
                        const LiveIntervals *LIS,
                        unsigned MaxInstNum =
                          std::numeric_limits<unsigned>::max()) {
  auto BB = Begin->getParent();
  OS << BB->getParent()->getName() << ":" << printMBBReference(*BB) << ' '
     << BB->getName() << ":\n";
  auto I = Begin;
  MaxInstNum = std::max(MaxInstNum, 1u);
  for (; I != End && MaxInstNum; ++I, --MaxInstNum) {
    if (!I->isDebugInstr() && LIS)
      OS << LIS->getInstructionIndex(*I);
    OS << '\t' << *I;
  }
  if (I != End) {
    OS << "\t...\n";
    I = std::prev(End);
    if (!I->isDebugInstr() && LIS)
      OS << LIS->getInstructionIndex(*I);
    OS << '\t' << *I;
  }
  if (End != BB->end()) { // print boundary inst if present
    OS << "----\n";
    if (LIS) OS << LIS->getInstructionIndex(*End) << '\t';
    OS << *End;
  }
}

LLVM_DUMP_METHOD
static void printLivenessInfo(raw_ostream &OS,
                              MachineBasicBlock::iterator Begin,
                              MachineBasicBlock::iterator End,
                              const LiveIntervals *LIS) {
  const auto BB = Begin->getParent();
  const auto &MRI = BB->getParent()->getRegInfo();

  const auto LiveIns = getLiveRegsBefore(*Begin, *LIS);
  OS << "LIn RP: ";
  getRegPressure(MRI, LiveIns).print(OS);

  const auto BottomMI = End == BB->end() ? std::prev(End) : End;
  const auto LiveOuts = getLiveRegsAfter(*BottomMI, *LIS);
  OS << "LOt RP: ";
  getRegPressure(MRI, LiveOuts).print(OS);

  const auto LiveThrough = getLiveThroughRegs(*Begin, *BottomMI, *LIS, MRI);
  OS << "LTr RP: ";
  getRegPressure(MRI, LiveThrough).print(OS);
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printRegions(raw_ostream &OS) const {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  for (const auto R : Regions) {
    OS << "Region to schedule ";
    printRegion(OS, R->Begin, R->End, LIS, 1);
    printLivenessInfo(OS, R->Begin, R->End, LIS);
    OS << "Max RP: ";
    R->MaxPressure.print(OS, &ST);
  }
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printSchedResult(raw_ostream &OS,
                                             const Region *R,
                                             const GCNRegPressure &RPBefore) const {
  OS << "\nAfter scheduling ";
  printRegion(OS, R->Begin, R->End, LIS);
  printSchedRP(OS, RPBefore, R->MaxPressure);
  OS << '\n';
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printSchedRP(raw_ostream &OS,
                                         const GCNRegPressure &Before,
                                         const GCNRegPressure &After) const {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  OS << "RP before: ";
  Before.print(OS, &ST);
  OS << "RP after:  ";
  After.print(OS, &ST);
}

template <typename Range>
LLVM_DUMP_METHOD
bool GCNIterativeScheduler::validateSchedule(const Region &R,
                                             const Range &Schedule) {
  std::vector<unsigned> NumPreds(SUnits.size());
  for (const auto &SU : SUnits)
    NumPreds[SU.NodeNum] = SU.NumPredsLeft;

  bool Res = true;
  unsigned NumInstr = 0;
  for (auto I = Schedule.begin(), E = Schedule.end(); I != E; ++I) {
    const auto *SU = *I;
    ++NumInstr;
    if (NumPreds[SU->NodeNum] != 0) {
      dbgs() << "ERROR: unsatisfied preds: " << NumPreds[SU->NodeNum] << ' ' << *SU->getInstr();
      for (auto &P : SU->Preds) {
        bool Met = false;
        for (auto J = Schedule.begin(); J != I; ++J) {
          if (P.getSUnit()->NodeNum == (*J)->NodeNum) {
            Met = true;
            break;
          }
        }
        if (!Met) {
          dbgs() << "  Missing pred: SU" << P.getSUnit()->NodeNum
                 << ' ' << *P.getSUnit()->getInstr();
        }
      }
      Res = false;
    }
    for (const auto &Succ : SU->Succs) {
      const auto &SuccSU = *Succ.getSUnit();
      if (!Succ.isWeak() && !SuccSU.isBoundaryNode()) {
        assert(NumPreds[SuccSU.NodeNum]);
        --NumPreds[SuccSU.NodeNum];
      }
    }
  }
  if (NumInstr != R.NumRegionInstrs) {
    dbgs() << "ERROR: schedule length mismatch: "
           << R.NumRegionInstrs << " before, "
           << NumInstr << " after\n";
    DenseMap<const MachineInstr*, unsigned> IC;
    for (const auto &MI : make_range(R.Begin, R.End))
      ++IC[&MI];
    for (const auto *SU : Schedule)
      ++IC[SU->getInstr()];
    for(const auto &P : IC)
      if (P.second == 1) {
        dbgs() << "Missing " << *P.first;
      }
    Res = false; // TODO: uncomment this
  }
  return Res;
}
#endif

std::string GCNIterativeScheduler::Region::getName(const LiveIntervals *LIS)
const {
  assert(LIS);
  std::string Name;
  raw_string_ostream O(Name);
  O << "dag." << getBB()->getParent()->getName() 
    << ".BB" << getBB()->getNumber() << '.'
    << LIS->getInstructionIndex(*Begin) << '-'
    << LIS->getInstructionIndex(*getLastMI());
  return O.str();
}

// DAG builder helper
class GCNIterativeScheduler::BuildDAG {
  GCNIterativeScheduler &Sch;
  SmallVector<SUnit *, 8> TopRoots;

  SmallVector<SUnit*, 8> BotRoots;
public:
  BuildDAG(const Region &R, GCNIterativeScheduler &_Sch)
    : Sch(_Sch) {
    auto BB = R.Begin->getParent();
    Sch.BaseClass::startBlock(BB);
    Sch.BaseClass::enterRegion(BB, R.Begin, R.End, R.NumRegionInstrs);

    Sch.buildSchedGraph(Sch.AA, nullptr, nullptr, nullptr,
                        /*TrackLaneMask*/true);
    Sch.Topo.InitDAGTopologicalSorting();
    Sch.findRootsAndBiasEdges(TopRoots, BotRoots);
  }

  ~BuildDAG() {
    Sch.BaseClass::exitRegion();
    Sch.BaseClass::finishBlock();
  }

  ArrayRef<const SUnit *> getTopRoots() const {
    return TopRoots;
  }
  ArrayRef<SUnit*> getBottomRoots() const {
    return BotRoots;
  }

  std::vector<const SUnit*>&& fixSchedule(std::vector<const SUnit*> &&Schedule) {
    auto Size = Sch.SUnits.size();
    if (Schedule.size() == Size)
      return std::move(Schedule);

    BitVector Scheduled(Size);
    for (auto *SU : Schedule)
      Scheduled.set(SU->NodeNum);

    Schedule.reserve(Size);
    for (auto I = Scheduled.find_first_unset(); I != -1;
              I = Scheduled.find_next_unset(I))
      Schedule.push_back(&Sch.SUnits[I]);

    return std::move(Schedule);
  }
};

class GCNIterativeScheduler::OverrideLegacyStrategy {
  GCNIterativeScheduler &Sch;
  Region &Rgn;
  std::unique_ptr<MachineSchedStrategy> SaveSchedImpl;
  GCNRegPressure SaveMaxRP;

public:
  OverrideLegacyStrategy(Region &R,
                         MachineSchedStrategy &OverrideStrategy,
                         GCNIterativeScheduler &_Sch)
    : Sch(_Sch)
    , Rgn(R)
    , SaveSchedImpl(std::move(_Sch.SchedImpl))
    , SaveMaxRP(R.MaxPressure) {
    Sch.SchedImpl.reset(&OverrideStrategy);
    auto BB = R.Begin->getParent();
    Sch.BaseClass::startBlock(BB);
    Sch.BaseClass::enterRegion(BB, R.Begin, R.End, R.NumRegionInstrs);
  }

  ~OverrideLegacyStrategy() {
    Sch.BaseClass::exitRegion();
    Sch.BaseClass::finishBlock();
    Sch.SchedImpl.release();
    Sch.SchedImpl = std::move(SaveSchedImpl);
  }

  void schedule() {
    assert(Sch.RegionBegin == Rgn.Begin && Sch.RegionEnd == Rgn.End);
    LLVM_DEBUG(dbgs() << "\nScheduling ";
               printRegion(dbgs(), Rgn.Begin, Rgn.End, Sch.LIS, 2));
    Sch.BaseClass::schedule();

    // Unfortunatelly placeDebugValues incorrectly modifies RegionEnd, restore
    Sch.RegionEnd = Rgn.End;
    //assert(Rgn.End == Sch.RegionEnd);
    Rgn.Begin = Sch.RegionBegin;
    Rgn.MaxPressure.clear();
  }

  void restoreOrder() {
    assert(Sch.RegionBegin == Rgn.Begin && Sch.RegionEnd == Rgn.End);
    // DAG SUnits are stored using original region's order
    // so just use SUnits as the restoring schedule
    Sch.scheduleRegion(Rgn, Sch.SUnits, SaveMaxRP);
  }
};

namespace {

// just a stub to make base class happy
class SchedStrategyStub : public MachineSchedStrategy {
public:
  bool shouldTrackPressure() const override { return false; }
  bool shouldTrackLaneMasks() const override { return false; }
  void initialize(ScheduleDAGMI *DAG) override {}
  SUnit *pickNode(bool &IsTopNode) override { return nullptr; }
  void schedNode(SUnit *SU, bool IsTopNode) override {}
  void releaseTopNode(SUnit *SU) override {}
  void releaseBottomNode(SUnit *SU) override {}
};

} // end anonymous namespace

GCNIterativeScheduler::GCNIterativeScheduler(MachineSchedContext *C,
                                             StrategyKind S)
  : BaseClass(C, llvm::make_unique<SchedStrategyStub>())
  , Context(C)
  , Strategy(S)
  , UPTracker(*LIS) {
}

// returns max pressure for a region
GCNRegPressure
GCNIterativeScheduler::getRegionPressure(MachineBasicBlock::iterator Begin,
                                         MachineBasicBlock::iterator End)
  const {
  // For the purpose of pressure tracking bottom inst of the region should
  // be also processed. End is either BB end, BB terminator inst or sched
  // boundary inst.
  auto const BBEnd = Begin->getParent()->end();
  auto const BottomMI = End == BBEnd ? std::prev(End) : End;

  // scheduleRegions walks bottom to top, so its likely we just get next
  // instruction to track
  auto AfterBottomMI = std::next(BottomMI);
  if (AfterBottomMI == BBEnd ||
      &*AfterBottomMI != UPTracker.getLastTrackedMI()) {
    UPTracker.reset(*BottomMI);
  } else {
    assert(UPTracker.isValid());
  }

  for (auto I = BottomMI; I != Begin; --I)
    UPTracker.recede(*I);

  UPTracker.recede(*Begin);

  assert(UPTracker.isValid() ||
         (dbgs() << "Tracked region ",
          printRegion(dbgs(), Begin, End, LIS), false));
  return UPTracker.moveMaxPressure();
}

// returns max pressure for a tentative schedule
template <typename Range> GCNRegPressure
GCNIterativeScheduler::getSchedulePressure(const Region &R,
                                           Range &&Schedule) const {
  auto const BBEnd = R.Begin->getParent()->end();
  GCNUpwardRPTracker RPTracker(*LIS);
  if (R.End != BBEnd) {
    // R.End points to the boundary instruction but the
    // schedule doesn't include it
    RPTracker.reset(*R.End);
    RPTracker.recede(*R.End);
  } else {
    // R.End doesn't point to the boundary instruction
    RPTracker.reset(*std::prev(BBEnd));
  }
  for (auto I = Schedule.end(), B = Schedule.begin(); I != B;) {
    RPTracker.recede(*getMachineInstr(*--I));
  }
  return RPTracker.moveMaxPressure();
}

void GCNIterativeScheduler::startBlock(MachineBasicBlock *BB) { // overriden
  removeM0Defs(*BB);
}

void GCNIterativeScheduler::finishBlock() { // overriden
  // do nothing
}

void GCNIterativeScheduler::enterRegion(MachineBasicBlock *BB, // overriden
                                        MachineBasicBlock::iterator Begin,
                                        MachineBasicBlock::iterator End,
                                        unsigned NumRegionInstrs) {
  //BaseClass::enterRegion(BB, Begin, End, NumRegionInstrs);
  if (NumRegionInstrs > 2) {
    Regions.push_back(
      new (Alloc.Allocate())
      Region { Begin, End, NumRegionInstrs,
               getRegionPressure(Begin, End), nullptr });
  }
}

void GCNIterativeScheduler::exitRegion() { // overriden
  // do nothing
}

void GCNIterativeScheduler::schedule() { // overriden
  // do nothing
#if 0  
  LLVM_DEBUG(printLivenessInfo(dbgs(), RegionBegin, RegionEnd, LIS);
             if (!Regions.empty() && Regions.back()->Begin == RegionBegin) {
               dbgs() << "Max RP: ";
               Regions.back()->MaxPressure.print(
                   dbgs(), &MF.getSubtarget<GCNSubtarget>());
             } dbgs()
             << '\n';);
#endif
}

void GCNIterativeScheduler::finalizeSchedule() { // overriden
  if (Regions.empty())
    return;
  switch (Strategy) {
  case SCHEDULE_MINREGONLY: scheduleMinReg(); break;
  case SCHEDULE_MINREGFORCED: scheduleMinReg(true); break;
  case SCHEDULE_LEGACYMAXOCCUPANCY: scheduleLegacyMaxOccupancy(); break;
  case SCHEDULE_ILP: scheduleILP(false); break;
  }
  for (auto &BB : MF)
    restoreM0Defs(BB);
  clearM0Map();
}

// Detach schedule from SUnits and interleave it with debug values.
// Returned schedule becomes independent of DAG state.
std::vector<MachineInstr*>
GCNIterativeScheduler::detachSchedule(ScheduleRef Schedule) const {
  std::vector<MachineInstr*> Res;
  Res.reserve(Schedule.size() * 2);

  if (FirstDbgValue)
    Res.push_back(FirstDbgValue);

  const auto DbgB = DbgValues.begin(), DbgE = DbgValues.end();
  for (auto SU : Schedule) {
    Res.push_back(SU->getInstr());
    const auto &D = std::find_if(DbgB, DbgE, [SU](decltype(*DbgB) &P) {
      return P.second == SU->getInstr();
    });
    if (D != DbgE)
      Res.push_back(D->first);
  }
  return Res;
}

void GCNIterativeScheduler::setBestSchedule(Region &R,
                                            ScheduleRef Schedule,
                                            const GCNRegPressure &MaxRP) {
  R.BestSchedule.reset(
    new TentativeSchedule{ detachSchedule(Schedule), MaxRP });
}

void GCNIterativeScheduler::scheduleBest(Region &R) {
  assert(R.BestSchedule.get() && "No schedule specified");
  scheduleRegion(R, R.BestSchedule->Schedule, R.BestSchedule->MaxPressure);
  R.BestSchedule.reset();
}

// minimal required region scheduler, works for ranges of SUnits*,
// SUnits or MachineIntrs*
template <typename Range>
void GCNIterativeScheduler::scheduleRegion(Region &R, Range &&Schedule,
                                           const GCNRegPressure &MaxRP) {
  assert(RegionBegin == R.Begin && RegionEnd == R.End);
  assert(LIS != nullptr);
#ifndef NDEBUG
  const auto SchedMaxRP = getSchedulePressure(R, Schedule);
#endif
  auto BB = R.Begin->getParent();
  auto Top = R.Begin;
  for (const auto &I : Schedule) {
    auto MI = getMachineInstr(I);
    if (MI != &*Top) {
      BB->remove(MI);
      BB->insert(Top, MI);
      if (!MI->isDebugInstr())
        LIS->handleMove(*MI, true);
    }
    if (!MI->isDebugInstr()) {
      // Reset read - undef flags and update them later.
      for (auto &Op : MI->operands())
        if (Op.isReg() && Op.isDef())
          Op.setIsUndef(false);

      RegisterOperands RegOpers;
      RegOpers.collect(*MI, *TRI, MRI, /*ShouldTrackLaneMasks*/true,
                                       /*IgnoreDead*/false);
      // Adjust liveness and add missing dead+read-undef flags.
      auto SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
      RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
    }
    Top = std::next(MI->getIterator());
  }
  RegionBegin = getMachineInstr(Schedule.front());

  // Schedule consisting of MachineInstr* is considered 'detached'
  // and already interleaved with debug values
  if (!std::is_same<decltype(*Schedule.begin()), MachineInstr*>::value) {
    placeDebugValues();
    // Unfortunatelly placeDebugValues incorrectly modifies RegionEnd, restore
    //assert(R.End == RegionEnd);
    RegionEnd = R.End;
  }

  R.Begin = RegionBegin;
  R.MaxPressure = MaxRP;

#ifndef NDEBUG
  const auto RegionMaxRP = getRegionPressure(R);
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
#endif
  assert((SchedMaxRP == RegionMaxRP && (MaxRP.empty() || SchedMaxRP == MaxRP))
  || (dbgs() << "Max RP mismatch!!!\n"
                "RP for schedule (calculated): ",
      SchedMaxRP.print(dbgs(), &ST),
      dbgs() << "RP for schedule (reported): ",
      MaxRP.print(dbgs(), &ST),
      dbgs() << "RP after scheduling: ",
      RegionMaxRP.print(dbgs(), &ST),
      false));
}

// Sort recorded regions by pressure - highest at the front
void GCNIterativeScheduler::sortRegionsByPressure(unsigned TargetOcc) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  llvm::sort(Regions, [&ST, TargetOcc](const Region *R1, const Region *R2) {
    return R2->MaxPressure.less(ST, R1->MaxPressure, TargetOcc);
  });
}

void GCNIterativeScheduler::computeDFSResult() {
  delete DFSResult;
  DFSResult = new SchedDFSResult2(8);
  DFSResult->resize(SUnits.size());
  DFSResult->compute(SUnits);
}

void GCNIterativeScheduler::removeM0Defs(MachineBasicBlock &BB) {
  DenseMap<int64_t, const MachineInstr*> M0Def;
  const MachineInstr *CurM0Def = nullptr;
  for (auto I = BB.begin(), E = BB.end(); I != E;) {
    auto &MI = *I++;
    if (MI.getOpcode() == AMDGPU::S_MOV_B32 &&
      MI.getOperand(0).getReg() == AMDGPU::M0) {
      const auto &Op1 = MI.getOperand(1);
      if (Op1.isImm()) {
        auto R = M0Def.try_emplace(Op1.getImm(), &MI);
        if (CurM0Def)
          R.second ? MI.removeFromParent() : MI.eraseFromParent();
        CurM0Def = R.first->second;
      }
      else
        CurM0Def = nullptr;
      continue;
    }

    if (CurM0Def && MI.findRegisterUseOperand(AMDGPU::M0))
      M0Map[&MI] = CurM0Def;
  }
}

void GCNIterativeScheduler::restoreM0Defs(MachineBasicBlock &BB) {
  const MachineInstr *CurM0Def = nullptr;
  for (auto &MI : BB) {
    if (MI.getOpcode() == AMDGPU::S_MOV_B32 &&
      MI.getOperand(0).getReg() == AMDGPU::M0 &&
      MI.getOperand(0).isImm()) {
      CurM0Def = &MI;
      continue;
    }

    auto I = M0Map.find(&MI);
    if (I != M0Map.end() && I->second != CurM0Def) {
      CurM0Def = I->second;
      BB.insert(&MI, MF.CloneMachineInstr(CurM0Def));
    }
  }
}

void GCNIterativeScheduler::clearM0Map() {
  /*for (auto &P : M0Map) {
    auto *MI = P.second;
  }*/
  // TODO: erase unused MIs
  M0Map.clear();
}

///////////////////////////////////////////////////////////////////////////////
// Legacy MaxOccupancy Strategy

// Tries to increase occupancy applying minreg scheduler for a sequence of
// most demanding regions. Obtained schedules are saved as BestSchedule for a
// region.
// TargetOcc is the best achievable occupancy for a kernel.
// Returns better occupancy on success or current occupancy on fail.
// BestSchedules aren't deleted on fail.
unsigned GCNIterativeScheduler::tryMaximizeOccupancy(unsigned TargetOcc) {
  // TODO: assert Regions are sorted descending by pressure
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  const auto Occ = Regions.front()->MaxPressure.getOccupancy(ST);
  LLVM_DEBUG(dbgs() << "Trying to improve occupancy, target = " << TargetOcc
                    << ", current = " << Occ << '\n');

  auto NewOcc = TargetOcc;
  for (auto R : Regions) {
    if (R->MaxPressure.getOccupancy(ST) >= NewOcc)
      break;

    LLVM_DEBUG(printRegion(dbgs(), R->Begin, R->End, LIS, 3);
               printLivenessInfo(dbgs(), R->Begin, R->End, LIS));

    BuildDAG DAG(*R, *this);
    const auto MinSchedule1 = DAG.fixSchedule(makeMinRegSchedule(DAG.getTopRoots(), *this));
    const auto MaxRP1 = getSchedulePressure(*R, MinSchedule1);
    LLVM_DEBUG(dbgs() << "Occupancy improvement attempt (minreg):\n";
               printSchedRP(dbgs(), R->MaxPressure, MaxRP1));

    if (DAG.getBottomRoots().size() > 1) {
      const auto MinSchedule2 = DAG.fixSchedule(
        makeMinRegSchedule2(DAG.getBottomRoots(),
                            getRegionLiveThrough(*R),
                            getRegionLiveOuts(*R),
                            *this));
      const auto MaxRP2 = getSchedulePressure(*R, MinSchedule2);
      LLVM_DEBUG(dbgs() << "Occupancy improvement attempt (minreg2):\n";
      printSchedRP(dbgs(), R->MaxPressure, MaxRP2));

      if (MaxRP2.less(ST, MaxRP1) && MaxRP2.getOccupancy(ST) >= NewOcc) {
        NewOcc = MaxRP2.getOccupancy(ST);
        setBestSchedule(*R, MinSchedule2, MaxRP2);
        continue;
      }
    }

    NewOcc = std::min(NewOcc, MaxRP1.getOccupancy(ST));
    if (NewOcc <= Occ)
      break;

    setBestSchedule(*R, MinSchedule1, MaxRP1);
  }
  LLVM_DEBUG(dbgs() << "New occupancy = " << NewOcc
                    << ", prev occupancy = " << Occ << '\n');
  if (NewOcc > Occ) {
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    MFI->increaseOccupancy(MF, NewOcc);
  }

  return std::max(NewOcc, Occ);
}

void GCNIterativeScheduler::scheduleLegacyMaxOccupancy(
  bool TryMaximizeOccupancy) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  auto TgtOcc = MFI->getMinAllowedOccupancy();

  sortRegionsByPressure(TgtOcc);
  auto Occ = Regions.front()->MaxPressure.getOccupancy(ST);

  if (TryMaximizeOccupancy && Occ < TgtOcc)
    Occ = tryMaximizeOccupancy(TgtOcc);

  // This is really weird but for some magic scheduling regions twice
  // gives performance improvement
  const int NumPasses = Occ < TgtOcc ? 2 : 1;

  TgtOcc = std::min(Occ, TgtOcc);
  LLVM_DEBUG(dbgs() << "Scheduling using default scheduler, "
                       "target occupancy = "
                    << TgtOcc << '\n');
  GCNMaxOccupancySchedStrategy LStrgy(Context);
  unsigned FinalOccupancy = std::min(Occ, MFI->getOccupancy());

  for (int I = 0; I < NumPasses; ++I) {
    // running first pass with TargetOccupancy = 0 mimics previous scheduling
    // approach and is a performance magic
    LStrgy.setTargetOccupancy(I == 0 ? 0 : TgtOcc);
    for (auto R : Regions) {
      OverrideLegacyStrategy Ovr(*R, LStrgy, *this);

      Ovr.schedule();
      const auto RP = getRegionPressure(*R);
      LLVM_DEBUG(printSchedRP(dbgs(), R->MaxPressure, RP));

      if (RP.getOccupancy(ST) < TgtOcc) {
        LLVM_DEBUG(dbgs() << "Didn't fit into target occupancy O" << TgtOcc);
        if (R->BestSchedule.get() &&
            R->BestSchedule->MaxPressure.getOccupancy(ST) >= TgtOcc) {
          LLVM_DEBUG(dbgs() << ", scheduling minimal register\n");
          scheduleBest(*R);
        } else {
          LLVM_DEBUG(dbgs() << ", restoring\n");
          Ovr.restoreOrder();
          assert(R->MaxPressure.getOccupancy(ST) >= TgtOcc);
        }
      }
      FinalOccupancy = std::min(FinalOccupancy, RP.getOccupancy(ST));
    }
  }
  MFI->limitOccupancy(FinalOccupancy);
}

///////////////////////////////////////////////////////////////////////////////
// Minimal Register Strategy

void GCNIterativeScheduler::scheduleMinReg(bool force) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const auto TgtOcc = MFI->getOccupancy();
  sortRegionsByPressure(TgtOcc);

  LLVM_DEBUG(printRegions(dbgs()));

  auto MaxPressure = Regions.front()->MaxPressure;
  for (auto *R : Regions) {
    LLVM_DEBUG(
      dbgs() << "Scheduling\n";
      printLivenessInfo(dbgs(), R->Begin, R->End, LIS);
      dbgs() << "RP: ";
      R->MaxPressure.print(dbgs(), &ST);
      dbgs() << '\n';
    );

    if (!force && R->MaxPressure.less(ST, MaxPressure, TgtOcc))
      break;

    BuildDAG DAG(*R, *this);
    LLVM_DEBUG(dbgs() << "\n=== Begin scheduling " << R->getName(LIS) << '\n');
    const auto MinSchedule = DAG.fixSchedule(makeMinRegSchedule2(DAG.getBottomRoots(),
                                                 getRegionLiveThrough(*R),
                                                 getRegionLiveOuts(*R),
                                                 *this));
    LLVM_DEBUG(dbgs() << "\n=== End scheduling " << R->getName(LIS) << '\n');
    assert(validateSchedule(*R, MinSchedule));

    const auto RPAfter = getSchedulePressure(*R, MinSchedule);
    LLVM_DEBUG(if (R->MaxPressure.less(ST, RPAfter, TgtOcc)) {
      dbgs() << "\nWarning: Pressure becomes worse after minreg!\n";
      printSchedRP(dbgs(), R->MaxPressure, RPAfter);
    });

    if (!force && MaxPressure.less(ST, RPAfter, TgtOcc))
      break;

    auto RPBefore = R->MaxPressure;
    scheduleRegion(*R, MinSchedule, RPAfter);
    LLVM_DEBUG(printSchedResult(dbgs(), R, RPBefore));

    MaxPressure = RPAfter;
  }
}

///////////////////////////////////////////////////////////////////////////////
// ILP scheduler port

void GCNIterativeScheduler::scheduleILP(
  bool TryMaximizeOccupancy) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  auto TgtOcc = MFI->getMinAllowedOccupancy();

  sortRegionsByPressure(TgtOcc);
  auto Occ = Regions.front()->MaxPressure.getOccupancy(ST);

  if (TryMaximizeOccupancy && Occ < TgtOcc)
    Occ = tryMaximizeOccupancy(TgtOcc);

  TgtOcc = std::min(Occ, TgtOcc);
  LLVM_DEBUG(dbgs() << "Scheduling using default scheduler, "
                       "target occupancy = "
                    << TgtOcc << '\n');

  unsigned FinalOccupancy = std::min(Occ, MFI->getOccupancy());
  for (auto R : Regions) {
    BuildDAG DAG(*R, *this);
    const auto ILPSchedule = makeGCNILPScheduler(DAG.getBottomRoots(), *this);

    const auto RP = getSchedulePressure(*R, ILPSchedule);
    LLVM_DEBUG(printSchedRP(dbgs(), R->MaxPressure, RP));

    if (RP.getOccupancy(ST) < TgtOcc) {
      LLVM_DEBUG(dbgs() << "Didn't fit into target occupancy O" << TgtOcc);
      if (R->BestSchedule.get() &&
        R->BestSchedule->MaxPressure.getOccupancy(ST) >= TgtOcc) {
        LLVM_DEBUG(dbgs() << ", scheduling minimal register\n");
        scheduleBest(*R);
      }
    }
    else {
      auto RPBefore = R->MaxPressure;
      scheduleRegion(*R, ILPSchedule, RP);
      LLVM_DEBUG(printSchedResult(dbgs(), R, RP));
      FinalOccupancy = std::min(FinalOccupancy, RP.getOccupancy(ST));
    }
  }
  MFI->limitOccupancy(FinalOccupancy);
}

#ifndef NDEBUG
namespace llvm {

template<>
struct GraphTraits<GCNIterativeScheduler*> : GraphTraits<ScheduleDAG*> {
};

template<>
class DOTGraphTraits<GCNIterativeScheduler*> : public DefaultDOTGraphTraits {
  
  mutable const GCNIterativeScheduler *DAG;

public:
  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getGraphName(const GCNIterativeScheduler *G) const {
    // this is the first method called with graph avaliable
    DAG = G;
    return G->MF.getName();
  }

  static bool renderGraphFromBottomUp() {
    return true;
  }

  bool isNodeHidden(const SUnit *Node) const {
    if (Node->Succs.size() > 127)
      return true;
    
    auto MI = Node->getInstr();
    if (MI->getOpcode() == AMDGPU::S_MOV_B32) {
      auto Op0 = MI->getOperand(0);
      if (Op0.isReg() && Op0.getReg() == AMDGPU::M0)
        return true;
    }

    auto DFS = DAG->getDFSResult();
    if (!DFS) return true;
   
    return !DFS->isInTreeOrDescendant(Node, 16) &&
           !DFS->isInTreeOrDescendant(Node, 17) &&
           !DFS->isInTreeOrDescendant(Node, 23) &&
           !DFS->isInTreeOrDescendant(Node, 24);
  }

  /// If you want to override the dot attributes printed for a particular
  /// edge, override this method.
  std::string getEdgeAttributes(const SUnit *Node,
                                SUnitIterator EI,
                                const ScheduleDAG *) const {
    if (EI.isArtificialDep())
      return "color=cyan,style=dashed";
    else if (EI.isCtrlDep())
      return "color=blue,style=dashed";

    auto DFS = DAG->getDFSResult();
    if (DFS) {
      auto NodeTreeID = DFS->getSubtreeID(Node);
      auto Pred = *EI;
      auto PredTreeID = DFS->getSubtreeID(Pred);
      if (NodeTreeID != PredTreeID)
        return DFS->isParentTree(NodeTreeID, PredTreeID) ?
          "color=green,style=bold":
          "color=orange,style=bold";
    }
    return "";
  }

  std::string getNodeLabel(const SUnit *SU, const ScheduleDAG *) const {
    std::string Str;
    raw_string_ostream O(Str);
    O << "SU:" << SU->NodeNum;
    if (const SchedDFSResult *DFS = DAG->getDFSResult())
      O << " I:" << DFS->getNumInstrs(SU);
    return O.str();
  }

  std::string getNodeDescription(const SUnit *SU, const ScheduleDAG *) const {
    return DAG->getGraphNodeLabel(SU);
  }

  std::string getNodeAttributes(const SUnit *N, const ScheduleDAG *) const {
    std::string Str;
    auto DFS = DAG->getDFSResult();
    if (DFS) {
      auto TopParent = DFS->getTopMostParentSubTreeID(N);
      auto Curr = DFS->getSubtreeID(N);
      Str += TopParent == Curr ? "style=\"bold,filled\"" : "style=\"filled\"";
      Str += ", fillcolor = \"#";
      Str += DOT::getColorString(TopParent);
      Str += '"';
    }
    return Str;
  }
};

static void fixFilename(std::string &Name) {
  for (auto &C : Name)
    if (C == ':')
      C = '.';
}

void GCNIterativeScheduler::writeGraph(StringRef Name) {
  auto Filename = std::string(Name) + ".dot";
  fixFilename(Filename);

  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, sys::fs::OpenFlags::F_Text);
  if (EC) {
    errs() << "Error opening " << Filename  << " file: " << EC.message() << '\n';
    return;
  }
  llvm::WriteGraph(FS, this, false, "Scheduling-Units Graph for " + Name);
}

void writeSubtreeGraph(const SchedDFSResult2 &R, StringRef Name) {
  auto Filename = std::string(Name) + ".subtrees.dot";
  fixFilename(Filename);

  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, sys::fs::OpenFlags::F_Text);
  if (EC) {
    errs() << "Error opening " << Filename << " file: " << EC.message() << '\n';
    return;
  }

  auto &O = FS;
  O << "digraph \"" << DOT::EscapeString(Name) << "\" {\n";

  for (size_t TreeID = 0, E = R.DFSTreeData.size(); TreeID < E; ++TreeID) {
    const auto &TD = R.DFSTreeData[TreeID];
    O << "\tSubtree" << TreeID
      << " [shape = record, style = \"filled\""
      << ", fillcolor = \"#" << DOT::getColorString(R.getTopMostParentSubTreeID(TreeID)) << '"'
      << ", label = \"{Subtree " << TreeID << "|SICount=" << TD.SubInstrCount
      << "}\"];\n";
    if (TD.ParentTreeID != SchedDFSResult::InvalidSubtreeID) {
      O << "\tSubtree" << TreeID << " -> "
        << "Subtree" << TD.ParentTreeID << "[color=green,style=bold];\n";
    }
  }

#if 0
  for (size_t TreeID = 0, E = R.SubtreeConnections.size(); TreeID < E; ++TreeID) {
    for (const auto &C : R.SubtreeConnections[TreeID]) {
      O << "\tSubtree" << TreeID << " -> "
        << "Subtree" << C.TreeID << "[color=orange,style=bold];\n";
    }
  }
#endif

  O << "}\n";
}

namespace {

  /// MachineScheduler runs after coalescing and before register allocation.
class PreRCMachineScheduler : public MachineSchedulerBase {
  public:
    PreRCMachineScheduler();

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    bool runOnMachineFunction(MachineFunction&) override;

    static char ID; // Class identification, replacement for typeinfo
  };

} // end anonymous namespace

char PreRCMachineScheduler::ID = 0;

void initializePreRCMachineSchedulerPass(PassRegistry&);

INITIALIZE_PASS_BEGIN(PreRCMachineScheduler, "prercmisched",
  "Pre Reg Coalescer Machine Instruction Scheduler", false, false)
  INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
  INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
  INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(PreRCMachineScheduler, "prercmisched",
  "Pre Reg Coalescer Machine Instruction Scheduler", false, false)

PreRCMachineScheduler::PreRCMachineScheduler() : MachineSchedulerBase(ID) {
  initializePreRCMachineSchedulerPass(*PassRegistry::getPassRegistry());
}

void PreRCMachineScheduler::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequiredID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool PreRCMachineScheduler::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(mf.getFunction()))
    return false;

  // Initialize the context of the pass.
  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  PassConfig = &getAnalysis<TargetPassConfig>();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  LIS = &getAnalysis<LiveIntervals>();

  LLVM_DEBUG(dbgs() << "Running minregonly\n");

  // Instantiate the selected scheduler for this target, function, and
  // optimization level.
  std::unique_ptr<ScheduleDAGInstrs> Scheduler(
    new GCNIterativeScheduler(this, GCNIterativeScheduler::SCHEDULE_MINREGONLY));
  scheduleRegions(*Scheduler, false);

  return true;
}

MachineFunctionPass* createPreRCMachineScheduler() {
  return new PreRCMachineScheduler();
}

} // end namespace llvm

#endif // NDEBUG
