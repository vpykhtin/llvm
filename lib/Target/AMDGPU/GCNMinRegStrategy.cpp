//===- GCNMinRegStrategy.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GCNIterativeScheduler.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/FileSystem.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

#if 1

namespace {

class GCNMinRegScheduler {
  struct Candidate : ilist_node<Candidate> {
    const SUnit *SU;
    int Priority;

    Candidate(const SUnit *SU_, int Priority_ = 0)
      : SU(SU_), Priority(Priority_) {}
  };

  SpecificBumpPtrAllocator<Candidate> Alloc;
  using Queue = simple_ilist<Candidate>;
  Queue RQ; // Ready queue

  std::vector<unsigned> NumPreds;

  bool isScheduled(const SUnit *SU) const {
    assert(!SU->isBoundaryNode());
    return NumPreds[SU->NodeNum] == std::numeric_limits<unsigned>::max();
  }

  void setIsScheduled(const SUnit *SU)  {
    assert(!SU->isBoundaryNode());
    NumPreds[SU->NodeNum] = std::numeric_limits<unsigned>::max();
  }

  unsigned getNumPreds(const SUnit *SU) const {
    assert(!SU->isBoundaryNode());
    assert(NumPreds[SU->NodeNum] != std::numeric_limits<unsigned>::max());
    return NumPreds[SU->NodeNum];
  }

  unsigned decNumPreds(const SUnit *SU) {
    assert(!SU->isBoundaryNode());
    assert(NumPreds[SU->NodeNum] != std::numeric_limits<unsigned>::max());
    return --NumPreds[SU->NodeNum];
  }

  void initNumPreds(const decltype(ScheduleDAG::SUnits) &SUnits);

  int getReadySuccessors(const SUnit *SU) const;
  int getNotReadySuccessors(const SUnit *SU) const;

  template <typename Calc>
  unsigned findMax(unsigned Num, Calc C);

  Candidate* pickCandidate();

  void bumpPredsPriority(const SUnit *SchedSU, int Priority);
  void releaseSuccessors(const SUnit* SU, int Priority);

public:
  std::vector<const SUnit*> schedule(ArrayRef<const SUnit*> TopRoots,
                                     const ScheduleDAG &DAG);
};

} // end anonymous namespace

void GCNMinRegScheduler::initNumPreds(const decltype(ScheduleDAG::SUnits) &SUnits) {
  NumPreds.resize(SUnits.size());
  for (unsigned I = 0; I < SUnits.size(); ++I)
    NumPreds[I] = SUnits[I].NumPredsLeft;
}

int GCNMinRegScheduler::getReadySuccessors(const SUnit *SU) const {
  unsigned NumSchedSuccs = 0;
  for (auto SDep : SU->Succs) {
    bool wouldBeScheduled = true;
    for (auto PDep : SDep.getSUnit()->Preds) {
      auto PSU = PDep.getSUnit();
      assert(!PSU->isBoundaryNode());
      if (PSU != SU && !isScheduled(PSU)) {
        wouldBeScheduled = false;
        break;
      }
    }
    NumSchedSuccs += wouldBeScheduled ? 1 : 0;
  }
  return NumSchedSuccs;
}

int GCNMinRegScheduler::getNotReadySuccessors(const SUnit *SU) const {
  return SU->Succs.size() - getReadySuccessors(SU);
}

template <typename Calc>
unsigned GCNMinRegScheduler::findMax(unsigned Num, Calc C) {
  assert(!RQ.empty() && Num <= RQ.size());

  using T = decltype(C(*RQ.begin())) ;

  T Max = std::numeric_limits<T>::min();
  unsigned NumMax = 0;
  for (auto I = RQ.begin(); Num; --Num) {
    T Cur = C(*I);
    if (Cur >= Max) {
      if (Cur > Max) {
        Max = Cur;
        NumMax = 1;
      } else
        ++NumMax;
      auto &Cand = *I++;
      RQ.remove(Cand);
      RQ.push_front(Cand);
      continue;
    }
    ++I;
  }
  return NumMax;
}

GCNMinRegScheduler::Candidate* GCNMinRegScheduler::pickCandidate() {
  do {
    unsigned Num = RQ.size();
    if (Num == 1) break;

    LLVM_DEBUG(dbgs() << "\nSelecting max priority candidates among " << Num
                      << '\n');
    Num = findMax(Num, [=](const Candidate &C) { return C.Priority; });
    if (Num == 1) break;

    LLVM_DEBUG(dbgs() << "\nSelecting min non-ready producing candidate among "
                      << Num << '\n');
    Num = findMax(Num, [=](const Candidate &C) {
      auto SU = C.SU;
      int Res = getNotReadySuccessors(SU);
      LLVM_DEBUG(dbgs() << "SU(" << SU->NodeNum << ") would left non-ready "
                        << Res << " successors, metric = " << -Res << '\n');
      return -Res;
    });
    if (Num == 1) break;

    LLVM_DEBUG(dbgs() << "\nSelecting most producing candidate among " << Num
                      << '\n');
    Num = findMax(Num, [=](const Candidate &C) {
      auto SU = C.SU;
      auto Res = getReadySuccessors(SU);
      LLVM_DEBUG(dbgs() << "SU(" << SU->NodeNum << ") would make ready " << Res
                        << " successors, metric = " << Res << '\n');
      return Res;
    });
    if (Num == 1) break;

    Num = Num ? Num : RQ.size();
    LLVM_DEBUG(
        dbgs()
        << "\nCan't find best candidate, selecting in program order among "
        << Num << '\n');
    Num = findMax(Num, [=](const Candidate &C) { return -(int64_t)C.SU->NodeNum; });
    assert(Num == 1);
  } while (false);

  return &RQ.front();
}

void GCNMinRegScheduler::bumpPredsPriority(const SUnit *SchedSU, int Priority) {
  SmallPtrSet<const SUnit*, 32> Set;
  for (const auto &S : SchedSU->Succs) {
    if (S.getSUnit()->isBoundaryNode() || isScheduled(S.getSUnit()) ||
        S.getKind() != SDep::Data)
      continue;
    for (const auto &P : S.getSUnit()->Preds) {
      auto PSU = P.getSUnit();
      assert(!PSU->isBoundaryNode());
      if (PSU != SchedSU && !isScheduled(PSU)) {
        Set.insert(PSU);
      }
    }
  }
  SmallVector<const SUnit*, 32> Worklist(Set.begin(), Set.end());
  while (!Worklist.empty()) {
    auto SU = Worklist.pop_back_val();
    assert(!SU->isBoundaryNode());
    for (const auto &P : SU->Preds) {
      if (!P.getSUnit()->isBoundaryNode() && !isScheduled(P.getSUnit()) &&
          Set.insert(P.getSUnit()).second)
        Worklist.push_back(P.getSUnit());
    }
  }
  LLVM_DEBUG(dbgs() << "Make the predecessors of SU(" << SchedSU->NodeNum
                    << ")'s non-ready successors of " << Priority
                    << " priority in ready queue: ");
  const auto SetEnd = Set.end();
  for (auto &C : RQ) {
    if (Set.find(C.SU) != SetEnd) {
      C.Priority = Priority;
      LLVM_DEBUG(dbgs() << " SU(" << C.SU->NodeNum << ')');
    }
  }
  LLVM_DEBUG(dbgs() << '\n');
}

void GCNMinRegScheduler::releaseSuccessors(const SUnit* SU, int Priority) {
  for (const auto &S : SU->Succs) {
    auto SuccSU = S.getSUnit();
    if (S.isWeak())
      continue;
    assert(SuccSU->isBoundaryNode() || getNumPreds(SuccSU) > 0);
    if (!SuccSU->isBoundaryNode() && decNumPreds(SuccSU) == 0)
      RQ.push_front(*new (Alloc.Allocate()) Candidate(SuccSU, Priority));
  }
}

std::vector<const SUnit*>
GCNMinRegScheduler::schedule(ArrayRef<const SUnit*> TopRoots,
                             const ScheduleDAG &DAG) {
  const auto &SUnits = DAG.SUnits;
  std::vector<const SUnit*> Schedule;
  Schedule.reserve(SUnits.size());

  initNumPreds(SUnits);

  int StepNo = 0;

  for (auto SU : TopRoots) {
    RQ.push_back(*new (Alloc.Allocate()) Candidate(SU, StepNo));
  }
  releaseSuccessors(&DAG.EntrySU, StepNo);

  while (!RQ.empty()) {
    LLVM_DEBUG(dbgs() << "\n=== Picking candidate, Step = " << StepNo
                      << "\n"
                         "Ready queue:";
               for (auto &C
                    : RQ) dbgs()
               << ' ' << C.SU->NodeNum << "(P" << C.Priority << ')';
               dbgs() << '\n';);

    auto C = pickCandidate();
    assert(C);
    RQ.remove(*C);
    auto SU = C->SU;
    LLVM_DEBUG(dbgs() << "Selected "; DAG.dumpNode(*SU));

    releaseSuccessors(SU, StepNo);
    Schedule.push_back(SU);
    setIsScheduled(SU);

    if (getReadySuccessors(SU) == 0)
      bumpPredsPriority(SU, StepNo);

    ++StepNo;
  }
  assert(SUnits.size() == Schedule.size());

  return Schedule;
}
#endif

namespace {

class GCNMinRegScheduler2 {

  struct Subgraph;
  class SGScheduler;
  class ReadyPredTracker;
  class SGRPTracker;
  class SGRPTracker2;

  struct LinkedSU : ilist_node<LinkedSU> {
    const SUnit * const SU;
    Subgraph *Parent = nullptr;
    bool hasExternalDataSuccs : 1; // true if this SU has data consumers in other SGs

    const SUnit *operator->() const { return SU; }
    const SUnit *operator->() { return SU; }

    LinkedSU(const SUnit &SU_) : SU(&SU_), hasExternalDataSuccs(false) {}
    LinkedSU(const LinkedSU&) = delete;
    LinkedSU(LinkedSU&&) = delete;

    unsigned getNodeNum() const {
      assert(!SU->isBoundaryNode());
      return SU->NodeNum;
    }

    bool dependsOn(const Subgraph &SG,
                   const GCNMinRegScheduler2 &LSUSource) const;

    void print(raw_ostream &OS) const {
      OS << "SU" << getNodeNum() << ": ";
      OS << *SU->getInstr();
    }
  };

  struct MergeSet {
    Subgraph *Center;

    struct TopoLess {
      bool operator()(const Subgraph *SG1, const Subgraph *SG2) const;
    };

    // Subgraphs to merge can be dependent on each other -
    // make the set topo sorted
    std::set<Subgraph*, TopoLess> Mergees;

    bool empty() const { return Mergees.empty(); }
    void clear() { Mergees.clear(); }

    bool operator<(const MergeSet &RHS) const {
      return Center != RHS.Center ?
        (Center < RHS.Center) :
        (Mergees < RHS.Mergees);
    }

    GCNRPTracker::LiveRegSet getLiveOutRegs() const {
      GCNRPTracker::LiveRegSet LiveOutRegs(Center->LiveOutRegs);
      for (auto *M : Mergees)
        LiveOutRegs += M->LiveOutRegs;
      return LiveOutRegs;
    }
  };

  typedef std::map<MergeSet, unsigned> MergeInfo;
  struct MergeChunk;

  struct Subgraph : ilist_node<Subgraph> {
    unsigned ID;
    simple_ilist<LinkedSU> List;
    DenseMap<Subgraph*, DenseSet<unsigned>> Preds;
    DenseMap<Subgraph*, DenseSet<unsigned>> Succs;
    GCNRPTracker::LiveRegSet LiveOutRegs;

    Subgraph(unsigned ID_,
             LinkedSU &Bot,
             const GCNRPTracker::LiveRegSet &RegionLiveOutRegs,
             const MachineRegisterInfo &MRI)
      : ID(ID_) {
      add(Bot, RegionLiveOutRegs, MRI);
    }

    void add(LinkedSU &LSU,
             const GCNRPTracker::LiveRegSet &RegionLiveOutRegs,
             const MachineRegisterInfo &MRI) {
      List.push_back(LSU);
      LSU.Parent = this;
      addLiveOut(LSU, RegionLiveOutRegs, MRI);
    }

    const SUnit *getBottomSU() const { return List.front().SU; }

    bool empty() const { return List.empty(); }

    bool isValid(const GCNMinRegScheduler2 &LSUSource) const;

    GCNRegPressure getLiveOutPressure() const {
      return getRegPressure(getBottomSU()->getInstr()->getMF()->getRegInfo(),
                            LiveOutRegs);
    }

    unsigned getNumLiveOut() const {
      unsigned NumLiveOut = 0;
      for (auto &LSU : List)
        NumLiveOut += LSU.hasExternalDataSuccs ? 1 : 0;
      return NumLiveOut;
    }

    unsigned getNumBottomRoots() const {
      unsigned NumBottomRoots = 0;
      for (auto &LSU : List) {
        if (LSU.hasExternalDataSuccs) continue;
        bool hasSucc = false;
        for (auto &Succ : LSU->Succs) {
          if (!Succ.getSUnit()->isBoundaryNode()) {
            hasSucc = true;
            break;
          }
        }
        if (!hasSucc)
          ++NumBottomRoots;
      }
      return NumBottomRoots;
    }

    bool dependsOn(const Subgraph &SG) const {
      return Preds.count(&SG) != 0;
    }

    bool hasIndirectPathTo(const Subgraph &ToSG) const {
      std::vector<const Subgraph*> Worklist;
      Worklist.reserve(32);
      for (auto &P : Preds) {
        if (P.first != &ToSG)
          Worklist.push_back(P.first);
      }
      DenseSet<const Subgraph*> Met;
      while (!Worklist.empty()) {
        auto *SG = Worklist.back();
        Worklist.pop_back();
        for (auto &P : SG->Preds) {
          if (P.first == &ToSG)
            return true;
          if (Met.insert(P.first).second)
            Worklist.push_back(P.first);
        }
      }
      return false;
    }

    std::set<Subgraph*> getDirectSuccs() const {
      std::set<Subgraph*> SuccSet;
      for (auto &P : Succs)
        if (P.first->Preds.size() == 1 ||
            !P.first->hasIndirectPathTo(*this))
          SuccSet.insert(P.first);
      return SuccSet;
    }

    template <typename Set>
    void insertMerges(MergeInfo &MergeGroups,
                      const Set &Tier,
                      const GCNMinRegScheduler2 &LSUSource);

    std::vector<MergeChunk> getMergeChunks(Subgraph *MergeTo,
                                           GCNMinRegScheduler2 &LSUSource);

    void dump(raw_ostream &O, GCNMinRegScheduler2 *LSUSource = nullptr) const;

#ifndef NDEBUG
    bool isValidMergeTo(Subgraph *MergeTo) const { 
      return !List.empty() && MergeTo->Succs.count(this) == 1;
    }
#endif

  private:
    void addLiveOut(LinkedSU &LSU,
                    const GCNRPTracker::LiveRegSet &RegionLiveOutRegs,
                    const MachineRegisterInfo &MRI);

    friend class ::GCNMinRegScheduler2;

    simple_ilist<LinkedSU> stripLiveOutDependentInstructions(
      GCNMinRegScheduler2 &LSUSource);

    template <typename Range>
    void mergeSchedule(Range &&R,
                       GCNMinRegScheduler2 &LSUSource);

    template <typename Range>
    void commitMerge(Range &&Mergees, const GCNMinRegScheduler2 &LSUSource);

    void patchDepsAfterMerge(Subgraph *Mergee);

    void rollbackMerge();
  };

  struct MergeChunk {
    LinkedSU* MergePoint = nullptr;
    decltype(Subgraph::List)::iterator Last;
    unsigned NumRegDeps;
  };

  const LiveIntervals &LIS;
  const MachineRegisterInfo &MRI;
  const GCNRPTracker::LiveRegSet &LiveThrRegs;
  const GCNRPTracker::LiveRegSet &LiveOutRegs;
  std::vector<LinkedSU> LSUStorage;

  std::vector<Subgraph> SGStorage;
  simple_ilist<Subgraph> Subgraphs;

  mutable std::vector<unsigned> UnitDepth;

  // merge sets claimed inefficient
  std::set<MergeSet> WastedMS;

  LinkedSU &getLSU(const SUnit *SU) {
    assert(!SU->isBoundaryNode());
    return LSUStorage[SU->NodeNum];
  }

  const LinkedSU &getLSU(const SUnit *SU) const {
    return const_cast<GCNMinRegScheduler2*>(this)->getLSU(SU);
  }

public:
  GCNMinRegScheduler2(ArrayRef<const SUnit*> BotRoots,
                      const GCNRPTracker::LiveRegSet &LTRegs,
                      const GCNRPTracker::LiveRegSet &LORegs,
                      const ScheduleDAGMI &DAG);

  std::vector<const SUnit*> schedule();

  void writeGraph(StringRef Name) const;

private:
  unsigned getSubgraphSuccNum(const LinkedSU &LSU) const;
  unsigned getExternalConsumersNum(const LinkedSU &LSU) const;
  unsigned getUnitDepth(const SUnit &SU) const;
  void setUnitDepthDirty(const SUnit &SU) const;

  void discoverSubgraphs(ArrayRef<const SUnit*> BotRoots);

  void scheduleSG(Subgraph &R);

  class AmbiguityMatrix;
  AmbiguityMatrix getAmbiguityMatrix(const MergeInfo &Merges) const;
  void disambiguateMerges(MergeInfo &Merges);
  void removeInefficientMerges(MergeInfo &Merges);

  MergeInfo getOneTierMerges();
  MergeInfo getMultiTierMerges();

  bool tryMerge(const MergeSet &MS);
  void merge();
  std::vector<const SUnit*> finalSchedule();

  static bool isExpanded(const Subgraph &R);
  void writeSubgraph(raw_ostream &O, const Subgraph &R) const;
  bool isEdgeHidden(const SUnit &SU, const SDep &Pred) const;
  void writeLSU(raw_ostream &O, const LinkedSU &LSU) const;
  void writeLinks(raw_ostream &O, const Subgraph &R) const;
};

bool GCNMinRegScheduler2::LinkedSU::dependsOn(const Subgraph &SG,
  const GCNMinRegScheduler2 &LSUSource) const {
  for (const auto &Pred : SU->Preds) {
    if (Pred.isWeak() || Pred.getSUnit()->isBoundaryNode())
      continue;
    if (LSUSource.getLSU(Pred.getSUnit()).Parent == &SG)
      return true;
  }
  return false;
}

// return true if SG1 should be placed before SG2
bool GCNMinRegScheduler2::MergeSet::TopoLess::operator()(const Subgraph *SG1, const Subgraph *SG2) const {
  if (SG2->dependsOn(*SG1))
    return true;
  if (SG1->dependsOn(*SG2))
    return false;
  return SG1 < SG2;
}

void GCNMinRegScheduler2::Subgraph::addLiveOut(LinkedSU &LSU,
  const GCNRPTracker::LiveRegSet &RegionLiveOutRegs,
  const MachineRegisterInfo &MRI) {
  for (const auto &MO : LSU.SU->getInstr()->defs()) {
    if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()) ||
      MO.isDead())
      continue;

    auto LO = RegionLiveOutRegs.find(MO.getReg());
    if (LO == RegionLiveOutRegs.end())
      continue;

    LaneBitmask LiveMask = LO->second & getDefRegMask(MO, MRI);
    if (LiveMask.any())
      LiveOutRegs[MO.getReg()] |= LiveMask;
  }
}

template <typename Range>
void GCNMinRegScheduler2::Subgraph::commitMerge(Range &&Mergees,
                                        const GCNMinRegScheduler2& LSUSource) {
  // set ownership on merged items
  for (auto &LSU : List)
    LSU.Parent = this;

  // patch hasExternalDataSuccs flag
  for (auto &LSU : List) {
    if (!LSU.hasExternalDataSuccs)
      continue;
    LSU.hasExternalDataSuccs = false;
    for (const auto &Succ : LSU->Succs) {
      auto *SU = Succ.getSUnit();
      if (SU->isBoundaryNode() || !Succ.isAssignedRegDep())
        continue;

      auto *SuccParent = LSUSource.getLSU(SU).Parent;
      if (SuccParent && SuccParent != this) {
        LSU.hasExternalDataSuccs = true;
        break;
      }
    }
  }

  for (auto *Mergee : Mergees) {
    // merge liveouts
    LiveOutRegs += Mergee->LiveOutRegs;

    patchDepsAfterMerge(Mergee);
  }

  assert(isValid(LSUSource));
}

void GCNMinRegScheduler2::Subgraph::patchDepsAfterMerge(Subgraph *Mergee) {
  assert(Succs.count(Mergee) == 1);
  for (auto &MrgPred : Mergee->Preds) {
    auto *MrgPredSG = MrgPred.first;
    MrgPredSG->Succs.erase(Mergee);
    if (MrgPredSG == this)
      continue;
    for (auto Reg : MrgPred.second) {
      Preds[MrgPredSG].insert(Reg); // insert MrgPredSG into SG Preds
      MrgPredSG->Succs[this].insert(Reg); // insert SG into MrgPredSG Succs
    }
  }
  for (auto &MrgSucc : Mergee->Succs) {
    auto *MrgSuccSG = MrgSucc.first;
    MrgSuccSG->Preds.erase(Mergee);
    for (auto Reg : MrgSucc.second) {
      Succs[MrgSuccSG].insert(Reg); // insert MrgSuccSG into SG Succs
      MrgSuccSG->Preds[this].insert(Reg); // insert SG into MrgSuccSG Preds
    }
  }
}

void GCNMinRegScheduler2::Subgraph::rollbackMerge() {
  // we assume List contains merged but not yet "committed" schedule that is
  // every LSU's Parent ins't updated yet and contains Subgraph the LSU came
  // from. Just put the LSU back to its Parent's List. The order is preserved
  // by the merging algorithm.
  for (auto I = List.begin(), E = List.end(); I != E;) {
    auto &LSU = *I++;
    if (LSU.Parent == this)
      continue;
    List.remove(LSU);
    LSU.Parent->List.push_back(LSU);
  }
}

class GCNMinRegScheduler2::SGRPTracker {
  GCNMinRegScheduler2 &LSUSource;
  Subgraph &SG;
  decltype(Subgraph::List.begin()) CurLSU;

  GCNUpwardRPTracker RPT;

  void stripDataPreds(LinkedSU &LSU) {
    for (const auto &Pred : LSU->Preds) {
      if (!Pred.isAssignedRegDep())
        continue;
      auto &PredLSU = LSUSource.getLSU(Pred.getSUnit());
      if (PredLSU.Parent != &SG)
        RPT.recedeDefsOnly(*PredLSU->getInstr());
    }
  }

public:
  SGRPTracker(GCNMinRegScheduler2 &LSUSource_,
              Subgraph &SG_)
    : LSUSource(LSUSource_)
    , SG(SG_)
    , RPT(LSUSource_.LIS) {
    RPT.addIgnoreRegs(LSUSource.LiveThrRegs);
  }

  GCNRegPressure getPressure() const { return RPT.getPressure(); }

  LinkedSU& getCur() const { return *CurLSU; }

  bool done() const { return CurLSU == SG.List.end(); }

  void reset() {
    CurLSU = SG.List.begin();
    RPT.reset(*(*CurLSU)->getInstr(), &SG.LiveOutRegs);
  }

  void recede() {
    assert(!done());
    RPT.recede(*(*CurLSU)->getInstr());
    stripDataPreds(*CurLSU);
    ++CurLSU;
  }
};


class GCNMinRegScheduler2::SGRPTracker2 {
  GCNMinRegScheduler2 &LSUSource;
  Subgraph &SG;
  decltype(Subgraph::List.begin()) CurLSU;
  unsigned NumRegs = 0, NumLiveOutRegs = 0;
  decltype(Subgraph::LiveOutRegs) LiveOutRegs;

  unsigned getNumLiveOut() const {
    unsigned Res = 0;
    auto &MRI = SG.List.front()->getInstr()->getMF()->getRegInfo();
    auto LiveOutRegs = SG.LiveOutRegs;
    for (auto &LSU : SG.List) {
      auto *MI = LSU->getInstr();
      if (MI->getNumOperands() < 1)
        continue;
      auto &Op0 = MI->getOperand(0);
      if (Op0.isReg() && Op0.isDef() &&
          TargetRegisterInfo::isVirtualRegister(Op0.getReg())) {
        auto I = LiveOutRegs.find(Op0.getReg());
        if (I != LiveOutRegs.end()) {
          ++Res;
          I->second &= ~getDefRegMask(Op0, MRI);
          if (I->second.none()) {
            LiveOutRegs.erase(I);
            if (LiveOutRegs.empty())
              break;
          }
        }
      }
    }
    return Res;
  }

public:
  SGRPTracker2(GCNMinRegScheduler2 &LSUSource_,
               Subgraph &SG_)
    : LSUSource(LSUSource_)
    , SG(SG_)
  {}

  unsigned getNumRegDeps() const { return NumRegs; }

  unsigned getNumRegLiveOut() const { return NumLiveOutRegs; }

  LinkedSU& getCur() const { return *CurLSU; }

  bool done() const { return CurLSU == SG.List.end(); }

  void reset() {
    CurLSU = SG.List.begin();
    NumRegs = NumLiveOutRegs = getNumLiveOut();
    LiveOutRegs = SG.LiveOutRegs;
  }

  void recede() {
    assert(!done());

    auto *MI = (*CurLSU)->getInstr();
    if (MI->getNumOperands() > 0) {
      auto &Op0 = MI->getOperand(0);
      if (Op0.isReg() && Op0.isDef() &&
        TargetRegisterInfo::isVirtualRegister(Op0.getReg())) {
        auto I = LiveOutRegs.find(Op0.getReg());
        if (I != LiveOutRegs.end()) {
          assert(NumLiveOutRegs > 0);
          --NumLiveOutRegs;
          auto &MRI = MI->getMF()->getRegInfo();
          I->second &= ~getDefRegMask(Op0, MRI);
          if (I->second.none())
            LiveOutRegs.erase(I);
        }
      }
    }

    for (const auto &Succ : (*CurLSU)->Succs) {
      if (!Succ.isAssignedRegDep())
        continue;
      const auto &SuccLSU = LSUSource.getLSU(Succ.getSUnit());
      if (SuccLSU.Parent == &SG)
        --NumRegs;
    }

    for (const auto &Pred : (*CurLSU)->Preds) {
      if (!Pred.isAssignedRegDep())
        continue;
      auto &PredLSU = LSUSource.getLSU(Pred.getSUnit());
      if (PredLSU.Parent == &SG)
        ++NumRegs;
    }

    ++CurLSU;
  }
};

std::vector<GCNMinRegScheduler2::MergeChunk>
GCNMinRegScheduler2::Subgraph::getMergeChunks(Subgraph *MergeTo,
                                              GCNMinRegScheduler2 &LSUSource) {
  struct Info {
    unsigned ExecOrder;
    unsigned NumDataSuccs;
  };
  DenseMap<LinkedSU*, Info> MergePointInfo; {
    unsigned ExecOrder = 0; // actually inverted order
    for (auto &LSU : MergeTo->List) {
      //if (!LSU.hasExternalDataSuccs)
      //  continue;

      unsigned NumDataSuccs = 0;
      bool HasCtrlSuccs = false;
      for (const auto &Succ : LSU->Succs) {
        if (Succ.isWeak() || Succ.getSUnit()->isBoundaryNode() ||
            LSUSource.getLSU(Succ.getSUnit()).Parent != this)
          continue;
        if (Succ.isAssignedRegDep())
          ++NumDataSuccs;
        else
          HasCtrlSuccs = true;
      }
      if (NumDataSuccs || HasCtrlSuccs)
        MergePointInfo[&LSU] = { ExecOrder++, NumDataSuccs };
    }
  }
  if (MergePointInfo.empty())
    return std::vector<GCNMinRegScheduler2::MergeChunk>();

  //const bool Dump = ID == 17;
  //const bool Dump = ID == 2;
  const bool Dump = false;

  std::vector<MergeChunk> Chunks(MergePointInfo.size());
  SGRPTracker2 RPT(LSUSource, *this);
  RPT.reset();

  unsigned CurMPIdx = 0;

  LLVM_DEBUG(if (Dump) {
    dbgs() << "NumRegs: " << RPT.getNumRegDeps()
           << ", LO Regs " << RPT.getNumRegLiveOut() << '\n';
  });

  while (!RPT.done()) {
    auto &CurLSU = RPT.getCur();
    RPT.recede();

    LLVM_DEBUG(if (Dump) {
      dbgs() << *CurLSU->getInstr();
      dbgs() << "NumRegs: " << RPT.getNumRegDeps()
             << ", LO Regs " << RPT.getNumRegLiveOut() << '\n';
    });

    auto LowestPredI = MergePointInfo.end();
    for (const auto &Pred : CurLSU->Preds) {
      if (Pred.isWeak() || Pred.getSUnit()->isBoundaryNode())
        continue;

      //if (!MPLSU.hasExternalDataSuccs)
      //  continue;

      auto &MPLSU = LSUSource.getLSU(Pred.getSUnit());
      if (MPLSU.Parent != MergeTo)
        continue;

      auto I = MergePointInfo.find(&MPLSU);
      assert(I != MergePointInfo.end());

      if (Pred.isAssignedRegDep()) {
        assert(I->second.NumDataSuccs);
        if (--I->second.NumDataSuccs != 0)
          continue;
      }

      if (LowestPredI == MergePointInfo.end() ||
          LowestPredI->second.ExecOrder > I->second.ExecOrder)
        LowestPredI = I;
    }

    if (LowestPredI != MergePointInfo.end()) {
      auto &MPInfo = LowestPredI->second;
      auto &C = Chunks[MPInfo.ExecOrder];
      C.MergePoint = LowestPredI->first;
      C.Last = CurLSU.getIterator();
      C.NumRegDeps = RPT.getNumRegDeps();

      LLVM_DEBUG(if (Dump) {
        dbgs() << "New merge point after: " << *(*C.MergePoint)->getInstr();
        dbgs() << "Best NumRegs: " << C.NumRegDeps << '\n';
      });

      if (MPInfo.ExecOrder < CurMPIdx) { // twist handling
        for (auto K = MPInfo.ExecOrder + 1; K <= CurMPIdx; ++K)
          Chunks[K].MergePoint = nullptr;
      }
      CurMPIdx = MPInfo.ExecOrder;
    } else if (RPT.getNumRegDeps() < Chunks[CurMPIdx].NumRegDeps) {
      Chunks[CurMPIdx].Last = CurLSU.getIterator();
      Chunks[CurMPIdx].NumRegDeps = RPT.getNumRegDeps();

      LLVM_DEBUG(if (Dump && Chunks[CurMPIdx].MergePoint) {
        dbgs() << "add after: " << *(*Chunks[CurMPIdx].MergePoint)->getInstr();
        dbgs() << "Best NumRegs: " << Chunks[CurMPIdx].NumRegDeps << '\n';
      });
    }
    LLVM_DEBUG(if (Dump) { dbgs() << '\n'; });
  }

  // shrink chunks
  std::vector<MergeChunk> ReadyChunks;
  ReadyChunks.reserve(MergePointInfo.size());
  for (auto &C : Chunks)
    if (C.MergePoint)
      ReadyChunks.push_back(C);

  return ReadyChunks;
}


simple_ilist<GCNMinRegScheduler2::LinkedSU> GCNMinRegScheduler2::Subgraph::
stripLiveOutDependentInstructions(GCNMinRegScheduler2 &LSUSource) {
  simple_ilist<LinkedSU> Res;
  auto &MRI = List.front()->getInstr()->getMF()->getRegInfo();
  auto LOR = LiveOutRegs;
  for (auto I = List.begin(), E = List.end(); I != E;) {
    auto &LSU = *I++;
    auto &MI = *LSU->getInstr();
    for (const auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;
      auto I = LOR.find(MO.getReg());
      if (I != LOR.end()) {
        I->second &= ~getDefRegMask(MO, MRI);
        if (I->second.none())
          LOR.erase(I);
      }
    }
    bool Strip = true;
    for (const auto &MO : MI.operands()) {
      if (!MO.isImm() && !(MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))) {
        Strip = false;
        break;
      }
      if (!MO.isReg() || !MO.isUse())
        continue;
      auto ULM = getUsedRegMask(MO, MRI, LSUSource.LIS);
      auto I = LOR.find(MO.getReg());
      if (I == LOR.end() || (I->second & ULM) != ULM) {
        Strip = false;
        break;
      }
    }
    if (Strip) {
      // check if this liveout is also used by consecuent instructions
      for (auto &S : LSU->Succs)
        if (!S.isWeak() && !S.getSUnit()->isBoundaryNode()) {
          Strip = false;
          break;
        }
    }
    if (Strip) {
      List.remove(LSU);
      Res.push_back(LSU);
    }
    if (LOR.empty())
      break;
  }
  return Res;
}

template <typename Range>
void GCNMinRegScheduler2::Subgraph::mergeSchedule(Range &&Mergees,
                                                  GCNMinRegScheduler2 &LSUSource) {
  for (auto *M : Mergees) {
    assert(M->isValidMergeTo(this));

    auto OutNumRegs = M->getNumBottomRoots();

    auto LODeps = M->stripLiveOutDependentInstructions(LSUSource);
    List.splice(List.begin(), LODeps);

    auto Chunks = M->getMergeChunks(this, LSUSource);
    auto Begin = M->List.begin();
    for (auto I = Chunks.begin(), E = Chunks.end(); I != E; ++I) {
      auto &P = *I;
      while (I != E && I->NumRegDeps > OutNumRegs)
        ++I;
      auto End = I == E ? M->List.end() : std::next(I->Last);
      List.splice(P.MergePoint->getIterator(), M->List, Begin, End);
      Begin = End;
      if (I == E)
        break;
    }

    if (!M->List.empty())
      List.splice(List.end(), M->List);
    assert(M->List.empty() || (M->dump(dbgs()), false));
  }

  LLVM_DEBUG(dbgs() << "\nMerge into SG" << ID << ":";
             for(auto *M : Mergees)
               dbgs() << " SG" << M->ID;
             dbgs() << '\n';
             dump(dbgs(), &LSUSource));
}

// returns the number of SU successors belonging to R
unsigned GCNMinRegScheduler2::getSubgraphSuccNum(const LinkedSU &LSU) const {
  unsigned NumSucc = 0;
  for (const auto &SDep : LSU.SU->Succs) {
    const auto *SuccSU = SDep.getSUnit();
    if (!SDep.isWeak() && !SuccSU->isBoundaryNode() &&
        getLSU(SuccSU).Parent == LSU.Parent)
      ++NumSucc;
  }
  return NumSucc;
}

/// Calculates the maximal path from the node to the exit.
unsigned GCNMinRegScheduler2::getUnitDepth(const SUnit &SU) const {
  assert(!SU.isBoundaryNode());
  if (UnitDepth[SU.NodeNum] != -1)
    return UnitDepth[SU.NodeNum];

  SmallVector<const SUnit*, 8> WorkList;
  WorkList.push_back(&SU);
  do {
    auto *Cur = WorkList.back();

    bool Done = true;
    unsigned MaxPredDepth = 0;
    for (const SDep &PredDep : Cur->Preds) {
      auto *PredSU = PredDep.getSUnit();
      if (PredSU->isBoundaryNode())
        continue;
      if (UnitDepth[PredSU->NodeNum] != -1)
        MaxPredDepth = std::max(MaxPredDepth, UnitDepth[PredSU->NodeNum] + 1);
      else {
        Done = false;
        WorkList.push_back(PredSU);
      }
    }

    if (Done) {
      WorkList.pop_back();
      if (MaxPredDepth != UnitDepth[Cur->NodeNum]) {
        setUnitDepthDirty(*Cur);
        UnitDepth[Cur->NodeNum] = MaxPredDepth;
      }
    }
  } while (!WorkList.empty());

  return UnitDepth[SU.NodeNum];
}

void GCNMinRegScheduler2::setUnitDepthDirty(const SUnit &SU) const {
  assert(!SU.isBoundaryNode());
  if (UnitDepth[SU.NodeNum] == -1)
    return;
  SmallVector<const SUnit*, 8> WorkList;
  WorkList.push_back(&SU);
  do {
    const SUnit *SU = WorkList.pop_back_val();
    UnitDepth[SU->NodeNum] = -1;
    for (const SDep &SuccDep : SU->Succs) {
      const auto *SuccSU = SuccDep.getSUnit();
      if (SuccSU->isBoundaryNode())
        continue;
      if (UnitDepth[SuccSU->NodeNum] != -1)
        WorkList.push_back(SuccSU);
    }
  } while (!WorkList.empty());
}


///////////////////////////////////////////////////////////////////////////////

GCNMinRegScheduler2::GCNMinRegScheduler2(
  ArrayRef<const SUnit*> BotRoots,
  const GCNRPTracker::LiveRegSet &LTRegs,
  const GCNRPTracker::LiveRegSet &LORegs,
  const ScheduleDAGMI &DAG)
  : LIS(*DAG.getLIS())
  , MRI(DAG.MRI)
  , LiveThrRegs(LTRegs)
  , LiveOutRegs(LORegs)
  , LSUStorage(DAG.SUnits.begin(), DAG.SUnits.end())
  , UnitDepth(DAG.SUnits.size(), -1) {

  discoverSubgraphs(BotRoots);
}

std::vector<const SUnit*> GCNMinRegScheduler2::schedule() {
  for (auto &R : Subgraphs) {
    scheduleSG(R);
    //DEBUG(R.dump(dbgs()));
  }

  merge();

  return finalSchedule();
}

/*
void GCNMinRegScheduler2::discoverSubgraph(Subgraph &SG) {
  std::vector<const SUnit*> Worklist;
  Worklist.push_back(SG.getBottomSU());

  do {
    auto *C = Worklist.back();
    Worklist.pop_back();

    for (auto &P : make_range(C->Preds.rbegin(), C->Preds.rend())) {
      if (P.isWeak())
        continue;
      auto &LSU = getLSU(P.getSUnit());
      if (!LSU.Parent) {
        SG.add(LSU, LiveOutRegs, MRI);
        Worklist.push_back(LSU.SU);
      } else if (LSU.Parent != &SG) { // cross edge detected
        auto Reg = P.isAssignedRegDep() ? P.getReg() : 0;
        SG.Preds[LSU.Parent].insert(Reg);
        LSU.Parent->Succs[&SG].insert(Reg);
        if (Reg)
          LSU.hasExternalDataSuccs = true;
      }
    }
  } while (!Worklist.empty());
}*/

void GCNMinRegScheduler2::discoverSubgraphs(ArrayRef<const SUnit*> BotRoots) {

  struct deepestFirst {
    const GCNMinRegScheduler2 &Sch;
    deepestFirst(const GCNMinRegScheduler2 &Sch_) : Sch(Sch_) {}

    bool operator()(const SUnit *SU1, const SUnit *SU2) {
      return Sch.getUnitDepth(*SU1) < Sch.getUnitDepth(*SU2);
    }
  };

  std::priority_queue<const SUnit*, std::vector<const SUnit*>, deepestFirst>
    Roots(BotRoots.begin(), BotRoots.end(), deepestFirst(*this));

  SGStorage.reserve(BotRoots.size());
  unsigned SubGraphID = 0;

  while (!Roots.empty()) {
    auto &LSU = getLSU(Roots.top());
    Roots.pop();

    SGStorage.emplace_back(SubGraphID++, LSU, LiveOutRegs, MRI);
    auto &SG = SGStorage.back();
    Subgraphs.push_back(SG);

    std::vector<const SUnit*> Worklist;
    Worklist.push_back(SG.getBottomSU());

    do {
      auto *C = Worklist.back();
      Worklist.pop_back();

      for (auto &P : make_range(C->Preds.rbegin(), C->Preds.rend())) {
        if (P.isWeak() || P.getSUnit()->isBoundaryNode())
          continue;

        auto &LSU = getLSU(P.getSUnit());
        if (!LSU.Parent) {
          SG.add(LSU, LiveOutRegs, MRI);
          Worklist.push_back(LSU.SU);
        }
        else if (LSU.Parent != &SG) { // cross edge detected
          auto Reg = P.isAssignedRegDep() ? P.getReg() : 0;
          SG.Preds[LSU.Parent].insert(Reg);
          LSU.Parent->Succs[&SG].insert(Reg);
          if (Reg)
            LSU.hasExternalDataSuccs = true;
        }
      }
    } while (!Worklist.empty());
  }
#ifndef NDEBUG
  for (auto &SG : Subgraphs)
    assert(SG.isValid(*this));
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Merging

class GCNMinRegScheduler2::AmbiguityMatrix {
  std::vector<std::set<unsigned>> Matrix;
  std::vector<unsigned> Erased;
public:
  AmbiguityMatrix(size_t N) : Matrix(N) {}

  void set(unsigned I, unsigned J) {
    Matrix[I].insert(J);
    Matrix[J].insert(I);
  }

  void erase(unsigned I) {
    auto &ToErase = Matrix[I];
    for (auto J : ToErase)
      Matrix[J].erase(I);
    ToErase.clear();
    Erased.push_back(I);
  }

  void eraseIncident(unsigned I) {
    auto ToErase(std::move(Matrix[I]));
    for (auto J : ToErase)
      erase(J);
  }

  iterator_range<decltype(Erased)::const_iterator> erased() const {
    return make_range(Erased.begin(), Erased.end());
  }

  const std::set<unsigned>& operator[](unsigned I) const { return Matrix[I]; }

  std::list<unsigned> getWorklist() const {
    std::list<unsigned> Worklist;
    for (unsigned I = 0, E = Matrix.size(); I < E; ++I)
      if (!Matrix[I].empty())
        Worklist.push_back(I);
    return Worklist;
  }
};

GCNMinRegScheduler2::AmbiguityMatrix
GCNMinRegScheduler2::getAmbiguityMatrix(const MergeInfo &Merges) const {
  std::vector<BitVector> SetMasks;
  SetMasks.reserve(Merges.size());
  for (auto &P : Merges) {
    SetMasks.emplace_back(BitVector(SGStorage.size()));
    auto &SetMask = SetMasks.back();
    SetMask.set(P.first.Center->ID);
    for (auto *M : P.first.Mergees)
      SetMask.set(M->ID);
  }
  AmbiguityMatrix AM(Merges.size());
  for (unsigned I = 0, E = Merges.size(); I < E; ++I)
    for (unsigned J = I + 1; J < E; ++J)
      if (SetMasks[I].anyCommon(SetMasks[J]))
        AM.set(I, J);
  return AM;
}

void GCNMinRegScheduler2::disambiguateMerges(MergeInfo &Merges) {
  std::vector<MergeInfo::iterator> IdxToMerge;
  IdxToMerge.reserve(Merges.size());
  for (auto I = Merges.begin(), E = Merges.end(); I != E; ++I)
    IdxToMerge.push_back(I);

  auto AM = getAmbiguityMatrix(Merges);
  auto Worklist = AM.getWorklist();
  while (!Worklist.empty()) {
    auto MostIncidentI = std::max_element(Worklist.begin(), Worklist.end(),
      [&AM](unsigned I, unsigned J) -> bool {
        return AM[I].size() < AM[J].size();
    });
    auto MostIncident = *MostIncidentI;
    Worklist.erase(MostIncidentI);

    auto &Incidents = AM[MostIncident];
    if (Incidents.empty())
      break;

    unsigned SumConsumedRegs = 0;
    for (auto J : Incidents)
      SumConsumedRegs += IdxToMerge[J]->second;

    if (IdxToMerge[MostIncident]->second <= SumConsumedRegs)
      AM.erase(MostIncident);
    else
      AM.eraseIncident(MostIncident);
  }

  for (auto I : AM.erased())
    Merges.erase(IdxToMerge[I]);
}

void GCNMinRegScheduler2::removeInefficientMerges(MergeInfo &Merges) {
  std::vector<unsigned> NumLiveOuts(SGStorage.size(), -1);
  for (auto I = Merges.begin(), E = Merges.end(); I != E;) {
    const auto This = I++;

    if (WastedMS.count(This->first)) {
      Merges.erase(This);
      continue;
    }

    unsigned NumLiveOutLeft = 0;
    unsigned NumRegsAdded = 0;
    for (auto *Mergee : This->first.Mergees) {
      auto &MergeeNumLiveOuts = NumLiveOuts[Mergee->ID];
      if (MergeeNumLiveOuts == -1)
        MergeeNumLiveOuts = Mergee->getNumLiveOut();
      NumLiveOutLeft += MergeeNumLiveOuts;
      NumRegsAdded += Mergee->getNumBottomRoots();
    }
    // if the number of registers left after the merge is more than consumed by
    // the merge - consider the merge inefficient
    if (NumRegsAdded >= This->second || NumLiveOutLeft > This->second)
      Merges.erase(This);
  }
}

template <typename Set>
void GCNMinRegScheduler2::Subgraph::insertMerges(MergeInfo &Merges,
                                        const Set &Tier,
                                        const GCNMinRegScheduler2 &LSUSource) {
  if (Succs.empty()) return;
  MergeSet MS;
  for (auto &LSU : List) {
    if (!LSU.hasExternalDataSuccs) continue;
    const auto &LSUSuccs = LSU.SU->Succs;
    for (const auto &Succ : LSUSuccs) {
      if (!Succ.isAssignedRegDep())
        continue;
      auto *SuccSG = LSUSource.getLSU(Succ.getSUnit()).Parent;
      if (SuccSG == this)
        continue;
      if (!Tier.count(SuccSG)) {
        MS.clear();
        break;
      } else
        MS.Mergees.insert(SuccSG);
    }

    if (!MS.empty()) {
      MS.Center = this;
      auto InsRes = Merges.emplace(std::move(MS), 1);
      if (!InsRes.second)
        ++InsRes.first->second;
      MS.clear();
    }
  }
}

GCNMinRegScheduler2::MergeInfo GCNMinRegScheduler2::getOneTierMerges() {
  MergeInfo Merges;
  for (auto &SG : Subgraphs) {
    if (!SG.Succs.empty())
      SG.insertMerges(Merges, SG.getDirectSuccs(), *this);
  }
  removeInefficientMerges(Merges);
  disambiguateMerges(Merges);
  return Merges;
}

GCNMinRegScheduler2::MergeInfo GCNMinRegScheduler2::getMultiTierMerges() {
  auto MaxI = std::max_element(Subgraphs.begin(), Subgraphs.end(),
    [=](const Subgraph &SG1, const Subgraph &SG2)->bool {
      return SG1.getNumLiveOut() < SG2.getNumLiveOut();
  });
  MergeInfo Merges;
  auto &Center = *MaxI;
  if (auto N = Center.getNumLiveOut()) {
    assert(!Center.Succs.empty());
    MergeSet MS = { &Center };
    for (auto &Succ : Center.Succs)
      MS.Mergees.insert(Succ.first);
    Merges.emplace(std::move(MS), N);
  }
  removeInefficientMerges(Merges);
  return Merges;
}

bool GCNMinRegScheduler2::tryMerge(const MergeSet &MS) {
  assert(!MS.Mergees.empty());
  auto LiveOutRegs = MS.getLiveOutRegs();
  GCNUpwardRPTracker RT(LIS);

  auto Range = make_range(MS.Mergees.rbegin(), MS.Mergees.rend());
  RT.reset(*(*Range.begin())->List.front()->getInstr(), &LiveOutRegs);
  for (auto *M : Range)
    for (auto &LSU : M->List)
      RT.recede(*LSU->getInstr());
  for (auto &LSU : MS.Center->List)
    RT.recede(*LSU->getInstr());
  const auto RPBefore = RT.moveMaxPressure();

  MS.Center->mergeSchedule(MS.Mergees, *this);

  RT.reset(*MS.Center->List.front()->getInstr(), &LiveOutRegs);
  for (auto &LSU : MS.Center->List)
    RT.recede(*LSU->getInstr());

  const auto RPAfter = RT.moveMaxPressure();

  LLVM_DEBUG(dbgs() << "RP before merge: "; RPBefore.print(dbgs());
             dbgs() << "RP after merge:  "; RPAfter.print(dbgs()));

  if (RPAfter.getVGPRNum() >= RPBefore.getVGPRNum()) {
    LLVM_DEBUG(dbgs() << "Unsuccessfull merge, rolling back\n");
    MS.Center->rollbackMerge();
    return false;
  }

  MS.Center->commitMerge(MS.Mergees, *this);
  for (auto *M : MS.Mergees)
    Subgraphs.remove(*M);
  return true;
}

void GCNMinRegScheduler2::merge() {
  LLVM_DEBUG(writeGraph("subdags_original.dot"));
  int I = 0;
  while(true) {
    auto Merges = getOneTierMerges();
    if (Merges.empty())
      break;
    for (auto &M : Merges) {
      if (!tryMerge(M.first))
        WastedMS.insert(M.first);
    }
    LLVM_DEBUG(writeGraph((Twine("subdags_merged") + Twine(I++) + ".dot").str()));
  }

#if 1
  while(true) {
    auto Merges = getMultiTierMerges();
    if (Merges.empty())
      break;
    for (auto &M : Merges) {
      if (!tryMerge(M.first))
        WastedMS.insert(M.first);
    }
    LLVM_DEBUG(writeGraph((Twine("subdags_merged") + Twine(I++) + ".dot").str()));
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Scheduling

class GCNMinRegScheduler2::SGScheduler {
  struct Unit { // schedule unit
    Unit(LinkedSU *LSU_, unsigned Priority_ = 0)
      : LSU(LSU_)
      , Priority(Priority_) {}
    LinkedSU *LSU;
    unsigned Priority;
  };
public:
  SGScheduler(Subgraph &SG_, GCNMinRegScheduler2 &LSUSource_)
    : SG(SG_)
    , LSUSource(LSUSource_)
    , SethiUllmanNumbers(calcSethiUllman())
    , NumSuccs(LSUSource_.LSUStorage.size())
    , Worklist(scheduleLater(*this), init()) {
  }

  void schedule() {
#ifndef NDEBUG
    auto PrevLen = SG.List.size();
#endif
    //LLVM_DEBUG(dbgs() << "Scheduling SG" << SG.ID <<'\n');
    SG.List.clear();
    unsigned StepNo = 0;
    while (!Worklist.empty()) {
      auto Unit = Worklist.top();
      Worklist.pop();
      SG.List.push_back(*Unit.LSU);
      releasePreds(*Unit.LSU, ++StepNo);
      //LLVM_DEBUG(dbgs() << "SUN: " << SethiUllmanNumbers[Unit.LSU->getNodeNum()]
      //                  << " " << *(*Unit.LSU)->getInstr());
    }
    assert(PrevLen == SG.List.size());
  }

private:
  Subgraph &SG;
  GCNMinRegScheduler2 &LSUSource;
  std::vector<unsigned> SethiUllmanNumbers;
  std::vector<unsigned> NumSuccs;

  struct scheduleLater {
    const SGScheduler &SGS;
    scheduleLater(const SGScheduler &SGS_) : SGS(SGS_) {}

    // return true if I1 should be scheduled later (in bottom-up order) than I2
    bool operator()(const Unit &I1, const Unit &I2) {
      if (I1.Priority != I2.Priority)
        return I1.Priority < I2.Priority;

      if (I1.LSU->hasExternalDataSuccs != I2.LSU->hasExternalDataSuccs)
        return I1.LSU->hasExternalDataSuccs > I2.LSU->hasExternalDataSuccs;
      else if (I1.LSU->hasExternalDataSuccs)
        return (*I1.LSU)->Succs.size() > (*I2.LSU)->Succs.size();

      const auto SUNum1 = SGS.SethiUllmanNumbers[I1.LSU->getNodeNum()];
      const auto SUNum2 = SGS.SethiUllmanNumbers[I2.LSU->getNodeNum()];
      if (SUNum1 != SUNum2)
        return SUNum1 > SUNum2;
      /*
      auto D1 = SGS.LSUSource.getUnitDepth(*LSU1->SU);
      auto D2 = SGS.LSUSource.getUnitDepth(*LSU2->SU);
      if (D1 != D2)
        return D1 < D2;
      */
      return false;
    }
  };
  std::priority_queue<Unit, std::vector<Unit>, scheduleLater> Worklist;

  std::vector<Unit> init() {
    std::vector<Unit> BotRoots;
    for (auto &LSU : SG.List) {
      if (auto NumSucc = LSUSource.getSubgraphSuccNum(LSU))
        NumSuccs[LSU.getNodeNum()] = NumSucc;
      else
        BotRoots.push_back(&LSU);
    }
    return BotRoots;
  }

  void releasePreds(LinkedSU &LSU, unsigned Priority) {
    for (auto &Pred : LSU->Preds) {
      const auto *PredSU = Pred.getSUnit();
      if (Pred.isWeak() || PredSU->isBoundaryNode())
        continue;

      auto &PredLSU = LSUSource.getLSU(PredSU);
      if (PredLSU.Parent != &SG)
        continue;

      assert(NumSuccs[PredLSU.getNodeNum()]);
      if (0 == --NumSuccs[PredLSU.getNodeNum()])
        Worklist.push(Unit(&PredLSU, Priority));
    }
  }

  std::vector<unsigned> calcSethiUllman() const {
    std::vector<unsigned> Res(LSUSource.LSUStorage.size());
    // start from the end of the list to reduce the depth of recursion in
    // calcSethiUllmanNumber
    for (auto &LSU : make_range(SG.List.rbegin(), SG.List.rend()))
      calcSethiUllmanNumber(LSU, Res);
    return Res;
  }

  static bool ignoreReg(unsigned Reg) {
    switch (Reg) {
    case AMDGPU::EXEC:
    case AMDGPU::M0: return true;
    default: break;
    }
    return false;
  }

  /// CalcNodeSethiUllmanNumber - Compute Sethi Ullman number.
  /// Smaller number is the higher priority.
  unsigned calcSethiUllmanNumber(LinkedSU &LSU,
    std::vector<unsigned> &SethiUllmanNumbers) const {
    unsigned &SethiUllmanNumber = SethiUllmanNumbers[LSU->NodeNum];
    if (SethiUllmanNumber != 0)
      return SethiUllmanNumber;

    unsigned Extra = 0;
    for (const SDep &Pred : LSU->Preds) {
      if (Pred.isCtrl() || Pred.getSUnit()->isBoundaryNode())
        continue;
      if (Pred.isAssignedRegDep() && ignoreReg(Pred.getReg()))
        continue;
      auto &PredLSU = LSUSource.getLSU(Pred.getSUnit());
      if (PredLSU.Parent != LSU.Parent)
        continue;
      auto PredSethiUllman = calcSethiUllmanNumber(PredLSU, SethiUllmanNumbers);
      if (PredSethiUllman > SethiUllmanNumber) {
        SethiUllmanNumber = PredSethiUllman;
        Extra = 0;
      }
      else if (PredSethiUllman == SethiUllmanNumber)
        ++Extra;
    }

    SethiUllmanNumber += Extra;

    if (SethiUllmanNumber == 0)
      SethiUllmanNumber = 1;

    return SethiUllmanNumber;
  }
};

void GCNMinRegScheduler2::scheduleSG(Subgraph &SG) {
  SGScheduler(SG, *this).schedule();
}

///////////////////////////////////////////////////////////////////////////////
// Final scheduling

std::vector<const SUnit*> GCNMinRegScheduler2::finalSchedule() {
  std::vector<Subgraph*> TopRoots;
  std::vector<unsigned> NumPreds(SGStorage.size());
  for (auto &SG : Subgraphs) {
    assert(!SG.empty());
    if (!SG.Preds.empty())
      NumPreds[SG.ID] = SG.Preds.size();
    else
      TopRoots.push_back(&SG);
  }
  
  std::vector<const SUnit*> Schedule;
  Schedule.reserve(LSUStorage.size());

  simple_ilist<LinkedSU> LODepList;

  for (auto *TopRoot : TopRoots) {
    std::vector<Subgraph*> Worklist;
    Worklist.push_back(TopRoot);
    do {
      std::sort(Worklist.begin(), Worklist.end(),
        [=](const Subgraph *R1, const Subgraph *R2) -> bool {
        return R1->List.size() > R2->List.size();
      });
      for (auto *SG : Worklist) {
        auto LODeps = SG->stripLiveOutDependentInstructions(*this);
        LODepList.splice(LODepList.begin(), LODeps);

        for (const auto &LSU : make_range(SG->List.rbegin(), SG->List.rend()))
          Schedule.push_back(LSU.SU);
      }

      std::vector<Subgraph*> NewWorklist;
      for (auto *SG : Worklist)
        for (auto &Succ : SG->Succs) {
          assert(NumPreds[Succ.first->ID]);
          if (0 == --NumPreds[Succ.first->ID])
            NewWorklist.push_back(Succ.first);
        }

      Worklist = std::move(NewWorklist);
    } while (!Worklist.empty());
  }

  for (const auto &LSU : make_range(LODepList.rbegin(), LODepList.rend()))
    Schedule.push_back(LSU.SU);

  return Schedule;
}

///////////////////////////////////////////////////////////////////////////////
// Dumping

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
bool GCNMinRegScheduler2::Subgraph::isValid(const GCNMinRegScheduler2 &LSUSource) const {
  bool Res = true;
  for (const auto &LSU : List) {

    if (LSU.Parent != this) {
      dbgs() << "ERROR: SG" << ID << " SU" << LSU->NodeNum
             << " has invalid parent\n";
      Res = false;
    }

    bool hasExternalDataSuccs = false;
    for (const auto &Succ : LSU->Succs) {
      if (Succ.getSUnit()->isBoundaryNode())
        continue;

      auto *SuccParent = LSUSource.getLSU(Succ.getSUnit()).Parent;
      if (SuccParent && SuccParent != this && Succs.count(SuccParent) == 0) {
        dbgs() << "ERROR: SG" << ID << " SU" << LSU->NodeNum
               << "'s successors's parent SG" << SuccParent->ID
               << " ins't found in the successors\n";
        Res = false;
      }

      if (SuccParent && SuccParent != this && Succ.isAssignedRegDep())
        hasExternalDataSuccs = true;
    }

    if (LSU.hasExternalDataSuccs != hasExternalDataSuccs) {
      dbgs() << "ERROR: SG" << ID << " SU" << LSU->NodeNum
             << (LSU.hasExternalDataSuccs ? "has" : " doesn't have")
             << "hasExternalDataSuccs flag\n";
      Res = false;
    }

    for (const auto &Pred : LSU->Preds) {
      if (Pred.getSUnit()->isBoundaryNode())
        continue;

      auto *PredParent = LSUSource.getLSU(Pred.getSUnit()).Parent;
      if (PredParent != this && Preds.count(PredParent) == 0) {
        dbgs() << "ERROR: SG" << ID << " SU" << LSU->NodeNum
               << "'s predecessors's parent SG" << PredParent->ID
               << " ins't found in the predecessors\n";
        Res = false;
      }
    }
  }
  return Res;
}
#endif

void GCNMinRegScheduler2::Subgraph::dump(raw_ostream &O,
                                        GCNMinRegScheduler2 *LSUSource) const {
  auto MergedLiveOutRegs = LiveOutRegs;
  DenseMap<const Subgraph*, unsigned> Level; {
    MergeSet MS;
    for (const auto &LSU : List)
      if (LSU.Parent != this)
        MS.Mergees.insert(LSU.Parent);
    // MS.Mergees is topo sorted
    for(auto *M: MS.Mergees) {
      unsigned MaxLevel = 0;
      for (const auto &Pred : M->Preds) {
        auto I = Level.find(Pred.first);
        if (I != Level.end())
          MaxLevel = std::max(MaxLevel, I->second);
      }
      Level[M] = MaxLevel + 1;
      MergedLiveOutRegs += M->LiveOutRegs;
    }
  }

  DenseMap<const LinkedSU*, unsigned> NumVGPRUsed;
  if (LSUSource) {
    GCNUpwardRPTracker RPT(LSUSource->LIS);
    RPT.reset(*List.front()->getInstr(), &MergedLiveOutRegs);
    for(const auto &LSU : List) {
      NumVGPRUsed[&LSU] = RPT.getPressure().getVGPRNum();
      RPT.recede(*LSU->getInstr());
    }
  }

  O << "Subgraph " << ID << '\n';
  for (const auto &LSU : make_range(List.rbegin(), List.rend())) {
    if (!NumVGPRUsed.empty())
      O << format("V%-4d ", NumVGPRUsed[&LSU], 4);

    if (LSU.Parent != this)
      O.indent(2 * Level[LSU.Parent]) << "SG" << LSU.Parent->ID << ": ";

    auto *MI = LSU->getInstr();
    if (MI->getNumOperands() > 1) {
      auto &Op0 = MI->getOperand(0);
      if (Op0.isReg() && Op0.isDef() && TargetRegisterInfo::isVirtualRegister(Op0.getReg())) {
        auto M = MergedLiveOutRegs.lookup(Op0.getReg());
        auto &MRI = MI->getMF()->getRegInfo();
        if ((M & getDefRegMask(Op0, MRI)).any())
          O << "> ";
      }
    }

    MI->print(O, true, false, false, false);
    O << ": SU" << LSU.getNodeNum() << '\n';
  }
  O << '\n';
}

unsigned GCNMinRegScheduler2::getExternalConsumersNum(const LinkedSU &LSU) const {
  DenseSet<const Subgraph*> Cons;
  for (auto &Succ : LSU.SU->Succs) {
    if (!Succ.isWeak() && Succ.isAssignedRegDep()) {
      auto *SuccR = getLSU(Succ.getSUnit()).Parent;
      if (SuccR != LSU.Parent)
        Cons.insert(SuccR);
    }
  }
  return Cons.size();
}

void GCNMinRegScheduler2::writeLSU(raw_ostream &O, const LinkedSU &LSU) const {
  const auto &SU = *LSU.SU;
  O << "\t\t";
  O << "SU" << SU.NodeNum
    << " [shape = record, style = \"filled\", rank="
    << getUnitDepth(SU);
  if (LSU.hasExternalDataSuccs)
    O << ", color = green";
  O << ", label = \"{SU" << SU.NodeNum;
  if (auto NumCons = getExternalConsumersNum(LSU))
    O << " C(" << NumCons << ')';
  O << '|';
  SU.getInstr()->print(O, /*IsStandalone=*/ false, /*SkipOpers=*/true);
  O << "}\"];\n";
}

static bool isSUHidden(const SUnit &SU) {
  if (SU.isBoundaryNode())
    return true;

  auto MI = SU.getInstr();
  if (MI->getOpcode() == AMDGPU::S_MOV_B32) {
    auto Op0 = MI->getOperand(0);
    if (Op0.isReg() && Op0.getReg() == AMDGPU::M0)
      return true;
  }
  return false;
}

static const char *getDepColor(SDep::Kind K) {
  switch (K) {
  case SDep::Anti: return "color=green";
  case SDep::Output: return "color=blue";
  case SDep::Order: return "color=red";
  default:;
  }
  return "";
}

static void writeSUtoPredSULink(raw_ostream &O, const SUnit &SU, const SDep &Pred) {
  O << "\tSU" << SU.NodeNum << " -> SU" << Pred.getSUnit()->NodeNum
    << '[' << getDepColor(Pred.getKind()) << "];\n";
}

void GCNMinRegScheduler2::writeSubgraph(raw_ostream &O, const Subgraph &SG) const {
  if (!isExpanded(SG)) {
    O << "\tSubgraph" << SG.ID
      << " [shape=record, style=\"filled\""
      << ", rank=" << getUnitDepth(*SG.getBottomSU())
      << ", fillcolor=\"#" << DOT::getColorString(SG.ID) << '"'
      << ", label = \"{{SG" << SG.ID
      << " (" << SG.List.size()
      << "MI)}|{LO=" << SG.getNumLiveOut()
      << " | BR=" << SG.getNumBottomRoots()
      << "}}\"];\n";
    return;
  }

  O << "\tsubgraph cluster_Subgraph" << SG.ID << " {\n";
  O << "\t\tlabel = \"Subgraph" << SG.ID << "\";\n";
  // write SUs
  for (const auto &LSU : SG.List) {
    if (!isSUHidden(*LSU.SU))
      writeLSU(O, LSU);
  }
  // write inner edges
  for (const auto &LSU : SG.List) {
    auto &SU = *LSU.SU;
    if (isSUHidden(SU))
      continue;
    for (const auto &Pred : SU.Preds) {
      if (Pred.isWeak() ||
          isSUHidden(*Pred.getSUnit()) ||
          getLSU(Pred.getSUnit()).Parent != &SG)
        continue;
      if (!isEdgeHidden(SU, Pred)) {
        O << '\t';
        writeSUtoPredSULink(O, SU, Pred);
      }
    }
  }
  O << "\t}\n";
}

bool GCNMinRegScheduler2::isEdgeHidden(const SUnit &SU, const SDep &Pred) const {
  return (Pred.getKind() == SDep::Order &&
    Pred.getSUnit()->Succs.size() > 5 &&
    abs((int)(SU.getHeight() - Pred.getSUnit()->getHeight())) > 10);
}

void GCNMinRegScheduler2::writeLinks(raw_ostream &O, const Subgraph &SG) const {
  const bool SGExpanded = isExpanded(SG);
  for (const auto &LSU : SG.List) {
    for (auto &Pred : LSU.SU->Preds) {
      const auto *PredSG = getLSU(Pred.getSUnit()).Parent;
      assert(PredSG != nullptr);
      if (Pred.isWeak() || PredSG == &SG)
        continue;
      const bool PredSGExpanded = isExpanded(*PredSG);
      if (!SGExpanded && !PredSGExpanded)
        continue;
      if (isSUHidden(*LSU.SU) ||
          isSUHidden(*Pred.getSUnit()) ||
          isEdgeHidden(*LSU.SU, Pred))
        continue;
      if (SGExpanded) {
        if (PredSGExpanded)
          writeSUtoPredSULink(O, *LSU.SU, Pred);
        else {
          O << "\tSU" << LSU.getNodeNum() << " -> Subgraph" << PredSG->ID;
          O << '[' << getDepColor(Pred.getKind()) << "];\n";
        }
      } else {
        assert(PredSGExpanded);
        O << "\tSubgraph" << SG.ID << " -> SU" << Pred.getSUnit()->NodeNum
          << '[' << getDepColor(Pred.getKind()) << "];\n";
      }
    }
  }
  if (SGExpanded)
    return;
  for (const auto &P : SG.Preds) {
    auto &PredSG = *P.first;
    if (isExpanded(PredSG))
      continue;
    unsigned HasCtrl = P.second.count(0) > 0 ? 1 : 0;
    if (HasCtrl) {
      O << "\tSubgraph" << SG.ID << " -> " << "Subgraph" << PredSG.ID
        << "[color=red];\n";
    }
    if ((P.second.size() - HasCtrl) > 0) {
      auto W = P.second.size() - HasCtrl;
      O << "\tSubgraph" << SG.ID << " -> " << "Subgraph" << PredSG.ID
        << "[weight=" << W << ",label=" << W << "];\n";
    }
  }
}


bool GCNMinRegScheduler2::isExpanded(const Subgraph &SG) {
  static const DenseSet<unsigned> Expand =
  {};
  //{ 0, 1, 4, 6 };
  // { 0, 1, 12, 16, 17, 19, 22, 23, 64};
  //{ 1, 2};

  return Expand.count(SG.ID) > 0 || (SG.List.size() <= 2);
}

void GCNMinRegScheduler2::writeGraph(StringRef Name) const {
  auto Filename = std::string(Name);

  std::error_code EC;
  raw_fd_ostream O(Filename, EC, sys::fs::OpenFlags::F_Text);
  if (EC) {
    errs() << "Error opening " << Filename << " file: " << EC.message() << '\n';
    return;
  }

  O << "digraph \"" << DOT::EscapeString(Name)
    << "\" {\n\trankdir=\"BT\";\n"
       "\tranksep=\"equally\";\n"
       "\tnewrank=\"true\";\n";
  for (auto &SG : Subgraphs) {
    writeSubgraph(O, SG);
  }
  for (auto &SG : Subgraphs) {
    if (!SG.Preds.empty())
      writeLinks(O, SG);
  }
  O << "}\n";
}

}

namespace llvm {

std::vector<const SUnit*> makeMinRegSchedule(ArrayRef<const SUnit*> TopRoots,
                                             const ScheduleDAG &DAG) {
  return GCNMinRegScheduler().schedule(TopRoots, DAG);
}

std::vector<const SUnit*> makeMinRegSchedule2(ArrayRef<const SUnit*> BotRoots,
                                              const GCNRPTracker::LiveRegSet &LiveThrRegs,
                                              const GCNRPTracker::LiveRegSet &LiveOutRegs,
                                              const ScheduleDAGMI &DAG) {
  return GCNMinRegScheduler2(BotRoots, LiveThrRegs, LiveOutRegs, DAG).schedule();
}

} // end namespace llvm
