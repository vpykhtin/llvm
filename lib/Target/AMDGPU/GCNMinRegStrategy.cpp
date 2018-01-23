//===- GCNMinRegStrategy.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GCNIterativeScheduler.h"
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
#include "llvm/Support/Filesystem.h"

#include <cassert>
#include <cstdint>
#include <limits>
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

  struct LinkedSU : ilist_node<LinkedSU> {
    const SUnit * const SU;
    Subgraph *Parent = nullptr;
    unsigned SGOrderIndex;

    LinkedSU(const SUnit &SU_) : SU(&SU_) {}

    unsigned getNodeNum() const {
      assert(!SU->isBoundaryNode());
      return SU->NodeNum;
    }

    bool dependsOn(const Subgraph &SG,
                   const GCNMinRegScheduler2 &LSUSource) const;
  };

  class Merge;
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

    unsigned getNumLiveOut() const {
      DenseSet<unsigned> Regs;
      for (auto &Succ : Succs)
        for (auto Reg : Succ.second)
          Regs.insert(Reg);
      return Regs.size();
    }

    typedef std::map<std::set<GCNMinRegScheduler2::Subgraph*>, unsigned> KillInfo;
    const KillInfo &getKills(const GCNMinRegScheduler2 &LSUSource) const;

    void updateOrderIndexes() {
      unsigned I = 0;
      for (auto &LSU : List)
        LSU.SGOrderIndex = I++;
    }

    void dump(raw_ostream &O) const;

  private:
    void addLiveOut(LinkedSU &LSU,
                    const GCNRPTracker::LiveRegSet &RegionLiveOutRegs,
                    const MachineRegisterInfo &MRI);

    friend class GCNMinRegScheduler2::Merge;

    void mergeSchedule(const DenseSet<Subgraph*> &Mergees,
                       GCNMinRegScheduler2 &LSUSource);

    template <typename Range>
    void commitMerge(const Range &Mergees);

    void patchDepsAfterMerge(Subgraph *Mergee);

    void rollbackMerge();

  private:
    mutable KillInfo Kills;
  };

  class Merge {
    Subgraph &SG;
    DenseSet<Subgraph*> Mergees;
    GCNMinRegScheduler2 &LSUSource;
    bool Cancelled = false;
  public:
    template <typename Range>
    Merge(Subgraph &SG_,
          Range &&Mergees_,
          GCNMinRegScheduler2 &LSUSource_)
      : SG(SG_)
      , LSUSource(LSUSource_) {
      for (auto *M : Mergees_)
        Mergees.insert(M);
      SG.mergeSchedule(Mergees, LSUSource);
    }
    ~Merge() {
      if (!Cancelled)
        commit();
    }
    void commit() {
      SG.commitMerge(Mergees);
      for (auto *M : Mergees)
        LSUSource.Subgraphs.remove(*M);
    }
    void rollback() {
      if (!Cancelled) {
        SG.rollbackMerge();
        Cancelled = true;
      }
    }
  };

  const LiveIntervals &LIS;
  const MachineRegisterInfo &MRI;
  const GCNRPTracker::LiveRegSet &LiveOutRegs;
  std::vector<LinkedSU> LSUStorage;

  std::vector<Subgraph> SGStorage;
  simple_ilist<Subgraph> Subgraphs;

  mutable std::vector<unsigned> UnitDepth;

  LinkedSU &getLSU(const SUnit *SU) {
    assert(!SU->isBoundaryNode());
    return LSUStorage[SU->NodeNum];
  }

  const LinkedSU &getLSU(const SUnit *SU) const {
    return const_cast<GCNMinRegScheduler2*>(this)->getLSU(SU);
  }

public:
  GCNMinRegScheduler2(ArrayRef<const SUnit*> BotRoots,
                      const GCNRPTracker::LiveRegSet &LORegs,
                      const ScheduleDAGMI &DAG);

  std::vector<const SUnit*> schedule();

  void writeGraph(StringRef Name) const;

private:
  unsigned getSubgraphSuccNum(const LinkedSU &LSU) const;
  unsigned getExternalConsumersNum(const LinkedSU &LSU) const;
  unsigned getUnitDepth(const SUnit &SU) const;
  void setUnitDepthDirty(const SUnit &SU) const;

  DenseMap<Subgraph*, LinkedSU*> getHighestSuccs(LinkedSU &LSU,
    const DenseSet<Subgraph*> &SGSet);

  void discoverSubgraph(Subgraph &R);
  void scheduleSubgraph(Subgraph &R);
  void merge();
  std::vector<const SUnit*> finalSchedule();

  static bool isExpanded(const Subgraph &R);
  void writeSubgraph(raw_ostream &O, const Subgraph &R) const;
  void writeExpandedSubgraph(raw_ostream &O, const Subgraph &R) const;
  bool isEdgeHidden(const SUnit &SU, const SDep &Pred) const;
  void writeLSU(raw_ostream &O, const LinkedSU &LSU) const;
  void writeLinks(raw_ostream &O, const Subgraph &R) const;
  void writeLinksExpanded(raw_ostream &O, const Subgraph &R) const;
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
void GCNMinRegScheduler2::Subgraph::commitMerge(const Range &Mergees) {
  // set ownership on merged items
  for (auto &LSU : List)
    LSU.Parent = this;

  for (auto *Mergee : Mergees) {
    // merge liveouts
    for (const auto &LO : Mergee->LiveOutRegs)
      LiveOutRegs[LO.first] |= LO.second;

    patchDepsAfterMerge(Mergee);
  }
}

void GCNMinRegScheduler2::Subgraph::patchDepsAfterMerge(Subgraph *Mergee) {
  assert(Succs.count(Mergee) == 1);
  for (auto &MrgPred : Mergee->Preds) {
    auto *MrgPredSG = MrgPred.first;
    MrgPredSG->Succs.erase(Mergee);
    MrgPredSG->Kills.clear();
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

}

DenseMap<GCNMinRegScheduler2::Subgraph*, GCNMinRegScheduler2::LinkedSU*>
GCNMinRegScheduler2::getHighestSuccs(LinkedSU &LSU,
                                     const DenseSet<Subgraph*> &SGSet) {
  DenseMap<Subgraph*, LinkedSU*> HighestSucc;
  for (const auto &Succ : LSU.SU->Succs) {
    if (Succ.isWeak() || Succ.getSUnit()->isBoundaryNode())
      continue;
    auto &SuccLSU = getLSU(Succ.getSUnit());
    if (LSU.Parent == SuccLSU.Parent || !SGSet.count(SuccLSU.Parent))
      continue;
    auto &H = HighestSucc[SuccLSU.Parent];
    if (!H || H->SGOrderIndex < SuccLSU.SGOrderIndex)
      H = &SuccLSU;
  }
  return HighestSucc;
}

void GCNMinRegScheduler2::Subgraph::mergeSchedule(const DenseSet<Subgraph*> &Mergees,
                                                  GCNMinRegScheduler2 &LSUSource) {
  DenseMap<Subgraph*, std::pair<GCNRegPressure, GCNUpwardRPTracker>> RP;

  updateOrderIndexes();
  for (auto *M : Mergees) {
    M->updateOrderIndexes();
    GCNUpwardRPTracker RPT(LSUSource.LIS);
    RPT.reset(*M->getBottomSU()->getInstr(), &M->LiveOutRegs);
    auto OutRP = RPT.getPressure();
    RP.insert(std::make_pair(M, std::make_pair(OutRP, std::move(RPT))));
  }

  for (auto &LSU : List) {
    if (LSU.SU->Succs.size() <= 1) // shortcut
      continue;

    const auto HighestSuccs = LSUSource.getHighestSuccs(LSU, Mergees);
    for (auto &P : HighestSuccs) {
      auto *Mergee = P.first;
      auto &MergeeList = Mergee->List;
      auto *MergeeHighestLSU = P.second;
      // check if LSU already merged
      if (MergeeList.empty() ||
          MergeeHighestLSU->SGOrderIndex < MergeeList.front().SGOrderIndex)
        continue;

      auto &RPInfo = RP.find(Mergee)->second;
      auto &RPT = RPInfo.second;
      auto I = std::next(MergeeHighestLSU->getIterator());
      for (auto &MrgLSU : make_range(MergeeList.begin(), I))
        RPT.recede(*MrgLSU.SU->getInstr());

      RPT.recede(*LSU.SU->getInstr());

      const auto &OutRP = RPInfo.first;
      auto CurRP = RPT.getPressure();
      while (I != MergeeList.end() &&
             !I->dependsOn(*this, LSUSource) &&
             (CurRP.getVGPRNum() > OutRP.getVGPRNum() ||
              CurRP.getSGPRNum() > OutRP.getSGPRNum())) {
        RPT.recede(*I->SU->getInstr());
        CurRP = RPT.getPressure();
        ++I;
      }

      List.splice(LSU.getIterator(), MergeeList, MergeeList.begin(), I);
    }
  }
  // merge in leftovers
  for (auto *MSG : Mergees) {
    if (!MSG->empty())
      List.splice(List.end(), MSG->List);
  }
}




const GCNMinRegScheduler2::Subgraph::KillInfo
&GCNMinRegScheduler2::Subgraph::getKills(const GCNMinRegScheduler2 &LSUSource) const {
  if (!Kills.empty())
    return Kills;

  for (auto &LSU : List) {
    const auto &Succs = LSU.SU->Succs;
    if (Succs.size() <= 1)
      continue;

    std::set<Subgraph*> Killers;
    for (const auto &Succ : Succs) {
      if (!Succ.isAssignedRegDep())
        continue;
      auto *SuccR = LSUSource.getLSU(Succ.getSUnit()).Parent;
      if (SuccR != this)
        Killers.insert(SuccR);
    }

    if (!Killers.empty())
      Kills[Killers]++;
  }
  return Kills;
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

///////////////////////////////////////////////////////////////////////////////

GCNMinRegScheduler2::GCNMinRegScheduler2(ArrayRef<const SUnit*> BotRoots,
  const GCNRPTracker::LiveRegSet &LORegs,
  const ScheduleDAGMI &DAG)
  : LIS(*DAG.getLIS())
  , MRI(DAG.MRI)
  , LiveOutRegs(LORegs)
  , LSUStorage(DAG.SUnits.begin(), DAG.SUnits.end())
  , UnitDepth(DAG.SUnits.size(), -1) {
  SGStorage.reserve(BotRoots.size());
  unsigned SubGraphID = 0;
  for (auto *SU : BotRoots) {
    SGStorage.emplace_back(SubGraphID++, getLSU(SU), LiveOutRegs, MRI);
    Subgraphs.push_back(SGStorage.back());
  }
}

std::vector<const SUnit*> GCNMinRegScheduler2::schedule() {
  // sort deepest first
  Subgraphs.sort([=](const Subgraph &R1, const Subgraph &R2) ->bool {
    return getUnitDepth(*R1.getBottomSU()) > getUnitDepth(*R2.getBottomSU());
  });

  for (auto &R : Subgraphs) {
    discoverSubgraph(R);
    scheduleSubgraph(R);
    R.updateOrderIndexes();
    //DEBUG(R.dump(dbgs()));
  }

  DEBUG(writeGraph("subdags_original.dot"));

  merge();

  DEBUG(writeGraph("subdags_merged.dot"));

  return finalSchedule();
}

void GCNMinRegScheduler2::discoverSubgraph(Subgraph &R) {
  std::vector<const SUnit*> Worklist;
  Worklist.push_back(R.getBottomSU());

  do {
    auto *C = Worklist.back();
    Worklist.pop_back();

    for (auto &P : make_range(C->Preds.rbegin(), C->Preds.rend())) {
      if (P.isWeak())
        continue;
      auto &LSU = getLSU(P.getSUnit());
      if (!LSU.Parent) {
        R.add(LSU, LiveOutRegs, MRI);
        Worklist.push_back(LSU.SU);
      } else if (LSU.Parent != &R) { // cross edge detected
        auto Reg = P.isAssignedRegDep() ? P.getReg() : 0;
        R.Preds[LSU.Parent].insert(Reg);
        LSU.Parent->Succs[&R].insert(Reg);
      }
    }
  } while (!Worklist.empty());
}

///////////////////////////////////////////////////////////////////////////////
// Merging

void GCNMinRegScheduler2::merge() {

  //Subgraphs.sort([=](const Subgraph &R1, const Subgraph &R2) ->bool {
  //  return true;
  //});

  bool Changed;
  do {
    Changed = false;
    for (auto &SG : Subgraphs) {

      for (const auto &K : SG.getKills(*this)) {
        unsigned InstrNum = 0;
        for (auto *SuccSG : K.first)
          InstrNum += SuccSG->List.size();
        assert(InstrNum > 0);
        if (InstrNum <= K.second * 2) {
          DEBUG(dbgs() << "Merging " << SG.ID << "\n");
          Merge(SG, K.first, *this);
          Changed = true;
          break;
        }
      }

    }
  } while (Changed);

#if 1
  Subgraph *SG1[] = { &SGStorage[1] };
  Merge(SGStorage[0], SG1, *this);
#endif

#if 1
  auto &SG0 = SGStorage[0];
  std::vector<Subgraph*> V;
  for (auto &Succ : SG0.Succs)
    if (Succ.first->ID==2)
      V.push_back(Succ.first);
  Merge(SG0, V, *this);
  SG0.dump(dbgs());
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Scheduling

void GCNMinRegScheduler2::scheduleSubgraph(Subgraph &R) {
  std::vector<unsigned> NumSuccs(LSUStorage.size());
  std::vector<LinkedSU*> Worklist;

  for (auto &LSU : R.List) {
    if (auto NumSucc = getSubgraphSuccNum(LSU))
      NumSuccs[LSU.getNodeNum()] = NumSucc;
    else
      Worklist.push_back(&LSU);
  }

  R.List.clear();
  while (!Worklist.empty()) {

    std::sort(Worklist.begin(), Worklist.end(),
      [=](const LinkedSU *LSU1, const LinkedSU *LSU2) {
        return LSU1->SU->Latency > LSU2->SU->Latency;
    });

    for (auto *LSU : Worklist) {
      R.List.push_back(*LSU);
    }

    std::vector<LinkedSU*> NewWorklist;
    for (const auto *LSU : Worklist)
      for (auto &Pred : LSU->SU->Preds) {
        const auto *PredSU = Pred.getSUnit();
        if (Pred.isWeak() || PredSU->isBoundaryNode())
          continue;

        auto &PredLSU = getLSU(PredSU);
        assert(PredLSU.Parent != &R || NumSuccs[PredLSU.getNodeNum()]);
        if (PredLSU.Parent == &R && 0 == --NumSuccs[PredLSU.getNodeNum()])
          NewWorklist.push_back(&PredLSU);
      }

    Worklist = std::move(NewWorklist);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Final scheduling

std::vector<const SUnit*> GCNMinRegScheduler2::finalSchedule() {
  SmallVector<Subgraph*, 4> TopRoots;
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

  for (auto *TopRoot : TopRoots) {
    std::vector<Subgraph*> Worklist;
    Worklist.push_back(TopRoot);
    do {
      std::sort(Worklist.begin(), Worklist.end(),
        [=](const Subgraph *R1, const Subgraph *R2) -> bool {
        return R1->List.size() > R2->List.size();
      });
      for (const auto *SG : Worklist)
        for (const auto &LSU : make_range(SG->List.rbegin(), SG->List.rend()))
          Schedule.push_back(LSU.SU);

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
  return Schedule;
}

///////////////////////////////////////////////////////////////////////////////
// Dumping

void GCNMinRegScheduler2::Subgraph::dump(raw_ostream &O) const {
  O << "Subgraph " << ID << '\n';
  for (const auto &LSU : make_range(List.rbegin(), List.rend())) {
    LSU.SU->getInstr()->print(O);
  }
  O << '\n';
}

//static const bool GraphScheduleMode = true;
static const bool GraphScheduleMode = false;

bool GCNMinRegScheduler2::isExpanded(const Subgraph &R) {
  static const DenseSet<unsigned> Expand =
    // { 0, 1, 2, 4, 6 };
    //{ 0, 1, 44, 45 };
     { 0, 1, 2 };
   //{ 0 };

  return Expand.count(R.ID) > 0 || (R.List.size() <= 2);
}

static bool isSUHidden(const SUnit &SU) {
  if (SU.isBoundaryNode())
    return true;

  //if (SU.Succs.size() > 100)
  //  return true;

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

void GCNMinRegScheduler2::writeSubgraph(raw_ostream &O, const Subgraph &R) const {
  auto SubGraphID = R.ID;
  O << "\tSubgraph" << SubGraphID
    << " [shape=record, style=\"filled\""
    << ", rank=" << getUnitDepth(*R.getBottomSU())
    << ", fillcolor=\"#" << DOT::getColorString(SubGraphID) << '"'
    << ", label = \"{Subgraph " << SubGraphID
    << "| IC=" << R.List.size()
    << ", LO=" << R.getNumLiveOut()
    << "}\"];\n";
}

void GCNMinRegScheduler2::writeLinks(raw_ostream &O, const Subgraph &R) const {
  for (const auto &P : R.Preds) {
    O << "\tSubgraph" << R.ID << " -> ";
    auto &PredR = *P.first;
    if (isExpanded(PredR))
      O << "SU" << PredR.getBottomSU()->NodeNum;
    else
      O << "Subgraph" << PredR.ID;
    O << "["
      << "weight=" << P.second.size()
      << ",label=" << P.second.size()
      << "];\n";
  }
}

bool GCNMinRegScheduler2::isEdgeHidden(const SUnit &SU, const SDep &Pred) const {
  //return abs((int)(getUnitDepth(SU) - getUnitDepth(*Pred.getSUnit()))) > 4;
  return (Pred.getKind() == SDep::Order &&
    Pred.getSUnit()->Succs.size() > 5 &&
    abs((int)(SU.getHeight() - Pred.getSUnit()->getHeight())) > 10);
}

static void writeSUtoPredSULink(raw_ostream &O, const SUnit &SU, const SDep &Pred) {
  O << "\tSU" << SU.NodeNum << " -> SU" << Pred.getSUnit()->NodeNum
    << '[' << getDepColor(Pred.getKind()) << "];\n";
}

void GCNMinRegScheduler2::writeLSU(raw_ostream &O, const LinkedSU &LSU) const {
  auto NumCons = getExternalConsumersNum(LSU);
  const auto &SU = *LSU.SU;
  O << "\t\t";
  O << "SU" << SU.NodeNum
    << " [shape = record, style = \"filled\", rank="
    << (GraphScheduleMode ? LSU.SGOrderIndex : getUnitDepth(SU));
  if (NumCons > 0)
    O << ", color = green";
  O << ", label = \"{SU" << SU.NodeNum;
  if (NumCons > 0)
    O << " C(" << NumCons << ')';
  O << '|';
  SU.getInstr()->print(O, /*SkipOpers=*/true);
  O << "}\"];\n";
}

void GCNMinRegScheduler2::writeExpandedSubgraph(raw_ostream &O, const Subgraph &R) const {
  auto SubGraphID = R.ID;
  O << "\tsubgraph cluster_Subgraph" << SubGraphID << " {\n";
  O << "\t\tlabel = \"Subgraph" << SubGraphID << "\";\n";
  // write SUs
  for (const auto &LSU : R.List) {
    if (!isSUHidden(*LSU.SU))
      writeLSU(O, LSU);
  }
  // write inner edges
  for (const auto &LSU : R.List) {
    auto &SU = *LSU.SU;
    if (isSUHidden(SU))
      continue;
    for (const auto &Pred : SU.Preds) {
      if (Pred.isWeak() ||
        isSUHidden(*Pred.getSUnit()) ||
        getLSU(Pred.getSUnit()).Parent != &R)
        continue;
      if (!isEdgeHidden(SU, Pred)) {
        O << '\t'; writeSUtoPredSULink(O, SU, Pred);
      }
    }
  }
  if (GraphScheduleMode) {
    // dot ingores rank between nodes without edges, so add an invisible edge
    // between consequent schedule units
    const LinkedSU *PrevLSU = nullptr;
    for (const auto &LSU : R.List) {
      if (isSUHidden(*LSU.SU))
        continue;
      if (PrevLSU) {
        O << "\t\tSU" << PrevLSU->getNodeNum()
          << "->SU" << LSU.getNodeNum()
          << " [style=invis];\n";
      }
      PrevLSU = &LSU;
    }
  }
  O << "\t}\n";
}

void GCNMinRegScheduler2::writeLinksExpanded(raw_ostream &O, const Subgraph &R) const {
  for (const auto &LSU : R.List) {
    if (isSUHidden(*LSU.SU))
      continue;
    for (auto &Pred : LSU.SU->Preds) {
      const auto *DepR = getLSU(Pred.getSUnit()).Parent;
      assert(DepR != nullptr);
      if (Pred.isWeak() ||
          isSUHidden(*Pred.getSUnit()) ||
          DepR == &R)
        continue;
      if (isExpanded(*DepR)) {
        if (!isEdgeHidden(*LSU.SU, Pred))
          writeSUtoPredSULink(O, *LSU.SU, Pred);
      } else {
        O << "\tSU" << LSU.getNodeNum() << " -> Subgraph" << DepR->ID;
        O << '[' << getDepColor(Pred.getKind()) << "];\n";
      }
    }
  }
}

void GCNMinRegScheduler2::writeGraph(StringRef Name) const {
  auto Filename = std::string(Name);

  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, sys::fs::OpenFlags::F_Text | sys::fs::OpenFlags::F_RW);
  if (EC) {
    errs() << "Error opening " << Filename << " file: " << EC.message() << '\n';
    return;
  }

  auto &O = FS;
  O << "digraph \"" << DOT::EscapeString(Name)
    << "\" {\n\trankdir=\"BT\";\n"
       "\tranksep=\"equally\";\n"
       "\tnewrank=\"true\";\n";
  // write subgraphs
  for (auto &R : Subgraphs) {
    if (isExpanded(R))
      writeExpandedSubgraph(O, R);
    else
      writeSubgraph(O, R);
  }
  // write links
  for (auto &R : Subgraphs) {
    if (R.Preds.empty())
      continue;
    if (isExpanded(R))
      writeLinksExpanded(O, R);
    else
      writeLinks(O, R);
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
                                              const GCNRPTracker::LiveRegSet &LiveOutRegs,
                                              const ScheduleDAGMI &DAG) {
  return GCNMinRegScheduler2(BotRoots, LiveOutRegs, DAG).schedule();
}

} // end namespace llvm
