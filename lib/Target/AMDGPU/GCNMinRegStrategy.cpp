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

#if 0
/// \brief Order nodes by the ILP metric.
struct MinOrder {
  const SchedDFSResult2 *DFSResult = nullptr;
  const BitVector *ScheduledTrees = nullptr;

  MinOrder() {}

  /// \brief Apply a less-than relation on node priority.
  ///
  /// (Return true if A comes after B in the Q.)
  bool operator()(const SUnit *A, const SUnit *B) const {
    unsigned SchedTreeA = DFSResult->getSubtreeID(A);
    unsigned SchedTreeB = DFSResult->getSubtreeID(B);
    if (SchedTreeA != SchedTreeB) {
      // Unscheduled trees have lower priority.
      if (ScheduledTrees->test(SchedTreeA) != ScheduledTrees->test(SchedTreeB))
        return ScheduledTrees->test(SchedTreeB);

      // Trees with shallower connections have have lower priority.
      if (DFSResult->getSubtreeLevel(SchedTreeA)
        != DFSResult->getSubtreeLevel(SchedTreeB)) {
        return DFSResult->getSubtreeLevel(SchedTreeA)
          < DFSResult->getSubtreeLevel(SchedTreeB);
      }
    }
    return DFSResult->getILP(A) < DFSResult->getILP(B);
  }
};

class GCNMinRegScheduler2  {
  std::unique_ptr<SchedDFSResult2> DFSResult;

  MinOrder Cmp;
  std::vector<const SUnit*> ReadyQ;

public:
  GCNMinRegScheduler2(const ScheduleDAG &DAG)
    : DFSResult(new SchedDFSResult2(8)) {
    DFSResult->resize(DAG.SUnits.size());
    DFSResult->compute(DAG.SUnits);
    Cmp.DFSResult = DFSResult.get();
  }

  void registerRoots() {
    // Restore the heap in ReadyQ with the updated DFS results.
    std::make_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
  }

  /// Implement MachineSchedStrategy interface.
  /// -----------------------------------------

  /// Callback to select the highest priority node from the ready Q.
  const SUnit *pickNode() {
    if (ReadyQ.empty()) return nullptr;
    std::pop_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
    auto *SU = ReadyQ.back();
    ReadyQ.pop_back();
    return SU;
  }

  /// \brief Scheduler callback to notify that a new subtree is scheduled.
  void scheduleTree(unsigned SubtreeID) {
    std::make_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
  }

  /// Callback after a node is scheduled. Mark a newly scheduled tree, notify
  /// DFSResults, and resort the priority Q.
  void schedNode(SUnit *SU, bool IsTopNode) {
    assert(!IsTopNode && "SchedDFSResult needs bottom-up");
  }

  void releaseBottomNode(SUnit *SU) {
    ReadyQ.push_back(SU);
    std::push_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
  }
};
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
    simple_ilist<LinkedSU> List;
    DenseMap<Root*, DenseSet<unsigned>> Preds;

    Root(LinkedSU &Bot) { add(Bot); }

    void add(LinkedSU &LSU) {
      List.push_back(LSU);
      LSU.Parent = this;
    }

    const SUnit *getBottomSU() const { return List.front().SU; }

    unsigned getID() const { return getBottomSU()->NodeNum; }

    void dump(raw_ostream &O) const;
  };

  std::vector<LinkedSU> LSUs;
  std::vector<Root> Roots;

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
    : LSUs(DAG.SUnits.begin(), DAG.SUnits.end()) {
    Roots.reserve(BotRoots.size());
    for (auto *SU : BotRoots) {
      Roots.emplace_back(getLSU(SU));
    }
  }

  std::vector<const SUnit*> schedule();

  void writeGraph(StringRef Name) const;

private:
  void discoverPseudoTree(Root &R);
  void schedulePseudoTree(Root &R);

  unsigned getNumSucc(const SUnit *SU, const Root &R) const;
};

std::vector<const SUnit*> GCNMinRegScheduler2::schedule() {
  // sort deepest first
  std::sort(Roots.begin(), Roots.end(),
    [=](const Root &R1, const Root &R2) ->bool {
    return R1.getBottomSU()->getDepth() > R2.getBottomSU()->getDepth();
  });

  for (auto &R : Roots) {
    discoverPseudoTree(R);
    //schedulePseudoTree(R);
    DEBUG(R.dump(dbgs()));
  }

  DEBUG(writeGraph("subtrees.dot"));

  std::vector<const SUnit*> Res;
  for (auto &LSU : LSUs)
    Res.push_back(LSU.SU);
  return Res;
}

void GCNMinRegScheduler2::discoverPseudoTree(Root &R) {
  std::vector<const SUnit*> Worklist;
  Worklist.push_back(R.getBottomSU());

  do {
    auto *C = Worklist.back();
    Worklist.pop_back();

    for (auto &P : make_range(C->Preds.rbegin(), C->Preds.rend())) {
      //if (!P.isAssignedRegDep()) continue;
      if (P.isWeak()) continue;

      auto &LSU = getLSU(P.getSUnit());
      if (!LSU.Parent) {
        R.add(LSU);
        Worklist.push_back(LSU.SU);
      } else if (LSU.Parent != &R) { // cross edge detected
        R.Preds[LSU.Parent].insert(P.isAssignedRegDep() ? P.getReg() : 0);
      }
    }
  } while (!Worklist.empty());
}

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

void GCNMinRegScheduler2::schedulePseudoTree(Root &R) {
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
      DEBUG(PredSU->getInstr()->print(dbgs()));
      R.List.push_back(PredLSU);
      DFS.follow(PredSU);
    }
    DFS.backtrack();
  } while (!DFS.isComplete());

  assert(R.List.size() == PrevSize);
}
///////////////////////////////////////////////////////////////////////////////
// Dumping

void GCNMinRegScheduler2::Root::dump(raw_ostream &O) const {
  O << "Subgraph " << getBottomSU()->NodeNum << '\n';
  for (const auto &LSU : make_range(List.rbegin(), List.rend())) {
    LSU.SU->getInstr()->print(O);
  }
  O << '\n';
}

void GCNMinRegScheduler2::writeGraph(StringRef Name) const {
  auto Filename = std::string(Name); // +".subtrees.dot";

  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, sys::fs::OpenFlags::F_Text | sys::fs::OpenFlags::F_RW);
  if (EC) {
    errs() << "Error opening " << Filename << " file: " << EC.message() << '\n';
    return;
  }

  auto &O = FS;
  O << "digraph \"" << DOT::EscapeString(Name) << "\" {\n";

  for (auto &R : Roots) {
    auto TreeID = R.getBottomSU()->NodeNum;
    O << "\tSubtree" << TreeID
      << " [shape = record, style = \"filled\""
      << ", fillcolor = \"#" << DOT::getColorString(TreeID) << '"'
      << ", label = \"{Subtree " << TreeID
      << "| InstrCount = " << R.List.size()
      << "}\"];\n";
    for(const auto &P: R.Preds) {
      O << "\tSubtree" << TreeID << " -> "
        << "Subtree" << P.first->getID()
        << "[" // color=green,style=bold
        << "weight=" << P.second.size()
        << ",label=" << P.second.size()
        << "];\n";
    }
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
