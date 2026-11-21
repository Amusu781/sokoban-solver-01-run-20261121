# Sokoban Solver

[![make.fun](https://make.fun/badge.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver, built for the make.fun bounty
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121).

## How it works

1. **Decompose.** A box pushed out of a room's door into the hallway can
   never be pushed back, so each room is an independent problem. The level
   is split into rooms by cutting the hallway row's dead squares out of the
   floor, and rooms are solved one at a time, stitched together with
   hallway walks.
2. **Search pushes, not moves.** A state is the sorted box squares plus
   the player's reachable region. The score counts pushes, so the search
   minimises pushes.
3. **A\* with an admissible heuristic.** The cheapest assignment of boxes
   to goals over grid distances (exact min-cost matching) never
   overestimates the pushes still needed, so the first solution found is
   push-optimal per room.
4. **Deadlock pruning.** Static dead squares (a box that can never be
   pulled back onto a goal) and frozen 2×2 blocks of walls and boxes are
   pruned before they are searched.
5. **Deterministic.** Fixed direction order, fixed tie-breaks, budgets
   counted in expansions — never the wall clock — so two runs write
   identical bytes.

## Run

```sh
node main.ts
```

`main.ts` reads levels from `MAKEFUN_INPUT_DIR` (default `/input`) and
writes `solution.txt` per instance under `MAKEFUN_OUTPUT_DIR`
(default `/output`), creating each output directory first.

## Licence

MIT — see [LICENSE](LICENSE).
