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

#if 0

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

  struct Root;

  struct LinkedSU : ilist_node<LinkedSU> {
    const SUnit * const SU;
    Root *Parent = nullptr;

    LinkedSU(const SUnit &SU_) : SU(&SU_) {}
  };

  struct Root {
    unsigned ID;
    simple_ilist<LinkedSU> List;
    DenseMap<Root*, DenseSet<unsigned>> Preds;
    DenseMap<Root*, DenseSet<unsigned>> Succs;

    Root(unsigned ID_, LinkedSU &Bot) : ID(ID_) { add(Bot); }

    void add(LinkedSU &LSU) {
      List.push_back(LSU);
      LSU.Parent = this;
    }

    const SUnit *getBottomSU() const { return List.front().SU; }

    unsigned getNumLiveOut() const {
      DenseSet<unsigned> Regs;
      for (auto &Succ : Succs)
        for (auto Reg : Succ.second)
          Regs.insert(Reg);
      return Regs.size();
    }

    void dump(raw_ostream &O) const;
  };

  std::vector<LinkedSU> LSUs;
  std::vector<Root> Roots;
  mutable std::vector<unsigned> UnitDepth;

  LinkedSU &getLSU(const SUnit *SU) {
    assert(!SU->isBoundaryNode());
    return LSUs[SU->NodeNum];
  }

  const LinkedSU &getLSU(const SUnit *SU) const {
    return const_cast<GCNMinRegScheduler2*>(this)->getLSU(SU);
  }

public:
  GCNMinRegScheduler2(ArrayRef<const SUnit*> BotRoots,
                      const ScheduleDAG &DAG)
    : LSUs(DAG.SUnits.begin(), DAG.SUnits.end())
    , UnitDepth(DAG.SUnits.size(), -1) {
    Roots.reserve(BotRoots.size());
    unsigned SubGraphID = 0;
    for (auto *SU : BotRoots) {
      Roots.emplace_back(SubGraphID++, getLSU(SU));
    }
  }

  std::vector<const SUnit*> schedule();

  void writeGraph(StringRef Name) const;

private:
  unsigned getNumSucc(const SUnit *SU, const Root &R) const;
  unsigned getConsumersNum(const LinkedSU &LSU) const;
  unsigned getUnitDepth(const SUnit &SU) const;
  void setUnitDepthDirty(const SUnit &SU) const;

  void discoverSubgraph(Root &R);
  void scheduleSubgraph(Root &R);
  void merge();
  bool mergeSuccessors(Root &R);
  std::map<std::set<Root*>, unsigned> getKills(const Root &R);
  template <typename Range> void merge(Root &R, Range &&Succs);
  std::vector<const SUnit*> finalSchedule();



  static bool isExpanded(const Root &R);
  void writeSubgraph(raw_ostream &O, const Root &R) const;
  void writeExpandedSubgraph(raw_ostream &O, const Root &R) const;
  bool isEdgeHidden(const SUnit &SU, const SDep &Pred) const;
  void writeLSU(raw_ostream &O, const LinkedSU &LSU) const;
  void writeLinks(raw_ostream &O, const Root &R) const;
  void writeLinksExpanded(raw_ostream &O, const Root &R) const;
};


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

unsigned GCNMinRegScheduler2::getConsumersNum(const LinkedSU &LSU) const {
  DenseSet<const Root*> Cons;
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

std::vector<const SUnit*> GCNMinRegScheduler2::schedule() {
  // sort deepest first
  std::sort(Roots.begin(), Roots.end(),
    [=](const Root &R1, const Root &R2) ->bool {
    return getUnitDepth(*R1.getBottomSU()) > getUnitDepth(*R2.getBottomSU());
  });

  for (auto &R : Roots) {
    discoverSubgraph(R);
    scheduleSubgraph(R);
    DEBUG(R.dump(dbgs()));
  }

  DEBUG(writeGraph("subdags_original.dot"));

  merge();

  DEBUG(writeGraph("subdags_merged.dot"));

  return finalSchedule();
}

void GCNMinRegScheduler2::discoverSubgraph(Root &R) {
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
        R.add(LSU);
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
  BitVector Visited(Roots.size());
  std::vector<unsigned> NumSuccs(Roots.size());
  std::vector<Root*> Worklist;
  bool Changed;
  do {
    Worklist.clear();
    for (auto &R : Roots) {
      if (R.List.empty())
        continue;
      if (R.Succs.empty())
        Worklist.push_back(&R);
      else
        NumSuccs[R.ID] = R.Succs.size();
    }

    Changed = false;
    while (!Changed && !Worklist.empty()) {
      std::vector<Root*> NewWorklist;
      for (auto *R : Worklist)
        for (auto &Pred : R->Preds)
          if (0 == --NumSuccs[Pred.first->ID])
            NewWorklist.push_back(Pred.first);

      Worklist = std::move(NewWorklist);

      for (auto *R : Worklist) {
        if (Visited.test(R->ID))
          continue;
        if (!mergeSuccessors(*R)) {
          Visited.set(R->ID);
          continue;
        }
        // After the merge R may become dependent on already visited root.
        // If so - clear Visited flag for such predecessors so they could try
        // to merge this R and restart the whole process
        for (const auto &Pred : R->Preds)
          if (Visited.test(Pred.first->ID)) {
            Visited.reset(Pred.first->ID);
            Changed = true;
          }
      }
    }
  } while (Changed);
}

std::map<std::set<GCNMinRegScheduler2::Root*>, unsigned>
GCNMinRegScheduler2::getKills(const Root &R) {
  std::map<std::set<Root*>, unsigned> Kills;
  for (auto &LSU : make_range(++R.List.begin(), R.List.end())) {
    const auto &Succs = LSU.SU->Succs;
    if (Succs.size() == 1)
      continue;

    std::set<Root*> Killers;
    for (const auto &Succ : Succs) {
      if (!Succ.isAssignedRegDep())
        continue;
      auto *SuccR = getLSU(Succ.getSUnit()).Parent;
      if (SuccR != &R)
        Killers.insert(SuccR);
    }

    if (!Killers.empty())
      Kills[Killers]++;
  }
  return Kills;
}

bool GCNMinRegScheduler2::mergeSuccessors(Root &R) {
  bool Changed = false;
  for (const auto &K : getKills(R)) {
    unsigned InstrNum = 0;
    for(const auto *R : K.first)
      InstrNum += R->List.size();
    assert(InstrNum > 0);
    if (InstrNum <= K.second * 2) {
      merge(R, K.first);
      Changed = true;
    }
  }
  return Changed;
}

template <typename Range>
void GCNMinRegScheduler2::merge(Root &R, Range &&Succs) {
  for (auto *Succ : Succs) {
    for (auto &LSU : Succ->List)
      LSU.Parent = &R;
    R.List.splice(R.List.begin(), Succ->List);
    R.Succs.erase(Succ);
    for (auto &Pred : Succ->Preds) {
      if (Pred.first != &R)
        for(auto Reg : Pred.second)
          R.Preds[Pred.first].insert(Reg);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Scheduling

void GCNMinRegScheduler2::scheduleSubgraph(Root &R) {
}

#if 0
/// Manage the stack used by a reverse depth-first search over the DAG.
class SchedDAGReverseDFS {
  std::vector<std::pair<const SUnit *, SUnit::const_pred_iterator>> DFSStack;

public:
  bool isComplete() const { return DFSStack.empty(); }

  void follow(const SUnit *SU) {
    DFSStack.push_back(std::make_pair(SU, SU->Preds.begin()));
  }
  void advance() { ++DFSStack.back().second; }

  const SDep *backtrack() {
    DFSStack.pop_back();
    return DFSStack.empty() ? nullptr : std::prev(DFSStack.back().second);
  }

  const SUnit *getCurr() const { return DFSStack.back().first; }

  SUnit::const_pred_iterator getPred() const { return DFSStack.back().second; }

  SUnit::const_pred_iterator getPredEnd() const {
    return getCurr()->Preds.end();
  }
};

// returns the number of SU successors belonging to R
unsigned GCNMinRegScheduler2::getNumSucc(const SUnit *SU, const Root &R) const {
  assert(getLSU(SU).Parent == &R);
  unsigned NumSucc = 0;
  for (const auto &SDep : SU->Succs) {
    const auto *SuccSU = SDep.getSUnit();
    if (!SDep.isWeak() && !SuccSU->isBoundaryNode() && getLSU(SuccSU).Parent == &R)
      ++NumSucc;
  }
  return NumSucc;
}

void GCNMinRegScheduler2::scheduleSubgraph(Root &R) {
  if (R.List.size() < 2)
    return;
#ifndef NDEBUG
  auto PrevSize = R.List.size();
#endif
  std::vector<unsigned> NumSucc(LSUs.size());
  auto Tail = make_range(++R.List.begin(), R.List.end());
  for (auto &LSU : Tail)
    NumSucc[LSU.SU->NodeNum] = getNumSucc(LSU.SU, R);

  R.List.erase(Tail.begin(), Tail.end());

  SchedDAGReverseDFS DFS;
  DFS.follow(R.getBottomSU());
  DEBUG(R.getBottomSU()->getInstr()->print(dbgs()));
  do {
    // Traverse the leftmost path as far as possible.
    while (DFS.getPred() != DFS.getPredEnd()) {
      const auto &Pred = *DFS.getPred();
      const auto *PredSU = Pred.getSUnit();
      DFS.advance();
      if (PredSU->isBoundaryNode() /* ||Pred.isWeak()*/)
        continue;
      auto &PredLSU = getLSU(PredSU);
      if (PredLSU.Parent != &R || --NumSucc[PredSU->NodeNum])
        continue;
      R.List.push_back(PredLSU);
      DFS.follow(PredSU);
    }
    DFS.backtrack();
  } while (!DFS.isComplete());

  assert(R.List.size() == PrevSize);
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Final scheduling

std::vector<const SUnit*> GCNMinRegScheduler2::finalSchedule() {
  std::vector<unsigned> NumPreds(Roots.size());
  for (auto &R : Roots) {
    if (R.List.empty())
      continue;
    if (!R.Preds.empty())
      NumPreds[R.ID] = R.Preds.size();
  }
  
  std::vector<const SUnit*> Schedule;
  Schedule.reserve(LSUs.size());

  std::vector<Root*> Worklist;
  Worklist.push_back(&Roots[0]);
  do {
    std::sort(Worklist.begin(), Worklist.end(),
      [=](const Root *R1, const Root *R2) -> bool {
        return R1->List.size() > R2->List.size();
    });
    for (const auto *R : Worklist)
      for (const auto &LSU : make_range(R->List.rbegin(), R->List.rend()))
        Schedule.push_back(LSU.SU);

    std::vector<Root*> NewWorklist;
    for (auto *R : Worklist)
      for (auto &Succ : R->Succs)
        if (0 == --NumPreds[Succ.first->ID])
          NewWorklist.push_back(Succ.first);

    Worklist = std::move(NewWorklist);
  } while (!Worklist.empty());

  return Schedule;
}

///////////////////////////////////////////////////////////////////////////////
// Dumping

void GCNMinRegScheduler2::Root::dump(raw_ostream &O) const {
  O << "Subgraph " << ID << '\n';
  for (const auto &LSU : make_range(List.rbegin(), List.rend())) {
    LSU.SU->getInstr()->print(O);
  }
  O << '\n';
}

bool GCNMinRegScheduler2::isExpanded(const Root &R) {
  static const DenseSet<unsigned> Expand = 
     { 0, 1, 2, 4, 6 };
    //{ 0, 1, 44, 45 };
    // { };

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

void GCNMinRegScheduler2::writeSubgraph(raw_ostream &O, const Root &R) const {
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

void GCNMinRegScheduler2::writeLinks(raw_ostream &O, const Root &R) const {
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
  auto NumCons = getConsumersNum(LSU);
  const auto &SU = *LSU.SU;
  O << "\t\t";
  O << "SU" << SU.NodeNum
    << " [shape = record, style = \"filled\", rank=" << getUnitDepth(SU);
  if (NumCons > 0)
    O << ", color = green";
  O << ", label = \"{SU" << SU.NodeNum;
  if (NumCons > 0)
    O << " C(" << NumCons << ')';
  O << '|';
  SU.getInstr()->print(O, /*SkipOpers=*/true);
  O << "}\"];\n";
}

void GCNMinRegScheduler2::writeExpandedSubgraph(raw_ostream &O, const Root &R) const {
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
  O << "\t}\n";
}

void GCNMinRegScheduler2::writeLinksExpanded(raw_ostream &O, const Root &R) const {
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
        O << "\tSU" << LSU.SU->NodeNum << " -> Subgraph" << DepR->ID;
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
  for (auto &R : Roots) {
    if (isExpanded(R))
      writeExpandedSubgraph(O, R);
    else
      writeSubgraph(O, R);
  }
  // write links
  for (auto &R : Roots) {
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
                                             ArrayRef<const SUnit*> BotRoots,
                                             const ScheduleDAG &DAG) {
  return GCNMinRegScheduler2(BotRoots, DAG).schedule();
}

} // end namespace llvm
