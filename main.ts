// Sokoban solver — push-optimal per room.
//
// Strategy (decompose, then search in the cheap space):
//  1. Split the level into independent rooms: a box pushed out of a room's
//     door into the hallway can never be pushed back, so each room is solved
//     on its own and the rooms are stitched together with hallway walks.
//  2. Inside a room, search over PUSHES, not moves: a state is the sorted
//     box squares plus the player's reachable region (named by its lowest
//     square). The score counts pushes, so the search minimises pushes.
//  3. A* with an admissible heuristic: the cheapest assignment of boxes to
//     goals over grid distances (exact min-cost matching, n <= 4), which
//     never overestimates the pushes still needed.
//  4. Deadlock pruning: static dead squares (a box that can never be pulled
//     back onto a goal), and frozen 2x2 blocks of walls/boxes.
//  5. Deterministic throughout: fixed direction order, fixed tie-breaks,
//     budgets counted in expansions, never the wall clock, so two runs
//     write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right.
const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
] as const;

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares a box can ever stand on and still reach a goal: pull backwards
// from the goals. A box on a square outside this set is dead, so any state
// containing one is pruned. Computed on the board alone (other boxes ignored).
function liveSquares(board: Board): Uint8Array {
  const { w, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const step = d.dy * w + d.dx;
      const y = x - step; // the square the box is pulled from
      if (!wall[y] && !wall[y - step] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Walking (grid) distances between every pair of floor squares, ignoring
// boxes: BFS from each square that is a box start or a goal. Only the rows
// we need are computed, keyed by square index.
function distanceRows(board: Board, sources: number[]): Map<number, Int32Array> {
  const { w, wall } = board;
  const out = new Map<number, Int32Array>();
  for (const s of sources) {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[s] = 0;
    const queue = [s];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const d of DIRS) {
        const q = p + d.dy * w + d.dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.set(s, dist);
  }
  return out;
}

// Cheapest one-to-one assignment of boxes to goals over the distance rows
// (exact, by trying every permutation; n <= 4 here). Admissible: each push
// moves one box one square, so the pushes still needed are at least the
// grid distance each box must still travel, matched optimally.
function matchingHeuristic(
  boxes: number[],
  goals: number[],
  distRows: Map<number, Int32Array>,
): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const perm = goals.map((_, i) => i);
  let best = Infinity;
  const walk = (used: number, cost: number): void => {
    if (cost >= best) return;
    const k = used; // boxes[0..k) already assigned, via perm order below
    if (k === n) {
      best = cost;
      return;
    }
    for (let j = 0; j < n; j++) {
      if (perm[j]! < 0) continue;
      const g = goals[perm[j]!]!;
      const d = distRows.get(g)![boxes[k]!]!;
      if (d < 0) continue; // this box cannot reach that goal; try another
      const saved = perm[j]!;
      perm[j] = -1;
      walk(k + 1, cost + d);
      perm[j] = saved;
    }
  };
  // The recursion above assigns boxes in order using the perm array as the
  // pool of free goals; the first call needs perm intact.
  walk(0, 0);
  return best;
}

// Shortest walk (lower-case LURD) from `from` to `to` avoiding walls and
// occupied squares. BFS, fixed direction order, so ties break identically
// on every run. Throws if unreachable (callers only ask for reachable).
function shortestWalk(
  board: Board,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  const { w, wall } = board;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('unreachable push square');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface PushStep {
  from: number; // box square before the push
  to: number; // box square after the push
  dirIndex: number;
}

// A* over pushes for one (sub-)level. Returns the push sequence, or null
// when the expansion budget runs out / no solution exists.
function solvePushes(board: Board, maxExpansions: number): PushStep[] | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const isGoal = new Uint8Array(size);
  for (const g of goals) isGoal[g] = 1;
  const live = liveSquares(board);
  if (board.boxes.some((b) => !live[b])) return null;
  const distRows = distanceRows(board, [...new Set(goals)]); // one row per goal: dist[box] = box-to-goal grid distance

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size); // flood-fill stamps for the current region
  const markSeen = new Int32Array(size); // separate stamps for regionLow, so it never clobbers `seen`
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;

  // Flood the player's reachable region from `from` with `occupied` in the
  // way; returns the region's lowest square and marks `markSeen` with a stamp.
  const regionLow = (from: number): number => {
    markStamp++;
    const mark = markStamp;
    let low = from;
    let n = 0;
    stack[n++] = from;
    markSeen[from] = mark;
    while (n > 0) {
      const p = stack[--n]!;
      if (p < low) low = p;
      for (const d of DIRS) {
        const q = p + d.dy * w + d.dx;
        if (!wall[q] && !occupied[q] && markSeen[q] !== mark) {
          markSeen[q] = mark;
          stack[n++] = q;
        }
      }
    }
    return low;
  };

  // Whether the box just pushed to `t` is caught in a frozen 2x2 block:
  // four squares of walls/boxes, one of them a box off its goal.
  const frozenAt = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      if (a < 0 || a % w === w - 1) continue; // the 2x2 must sit inside one row pair
      const b = a + 1;
      const c = a + w;
      const d2 = c + 1;
      if (a < 0 || d2 >= size) continue;
      const filled = (i: number): boolean => wall[i] === 1 || occupied[i] === 1;
      if (!filled(a) || !filled(b) || !filled(c) || !filled(d2)) continue;
      const offGoal = (i: number): boolean => occupied[i] === 1 && !isGoal[i];
      if (offGoal(a) || offGoal(b) || offGoal(c) || offGoal(d2)) return true;
    }
    return false;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => isGoal[b]);
  const stateKey = (boxes: number[], low: number): string =>
    `${low}|${boxes.join(',')}`;

  // Open list: binary heap on [f, -g, insertion order] so ties prefer
  // deeper states, then earlier insertion — fully deterministic.
  type HeapEntry = [number, number, number, number]; // f, -g, order, state
  const heap: HeapEntry[] = [];
  const heapLess = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const boxesOf: number[][] = [board.boxes.slice()];
  const playerOf: number[] = [board.player];
  const gOf: number[] = [0];
  const parentOf: number[] = [-1];
  const pushOf: (PushStep | null)[] = [null];
  const bestG = new Map<string, number>();

  for (const b of board.boxes) occupied[b] = 1;
  const startLow = regionLow(board.player);
  for (const b of board.boxes) occupied[b] = 0;
  bestG.set(stateKey(board.boxes, startLow), 0);
  const h0 = matchingHeuristic(board.boxes, goals, distRows);
  if (h0 === Infinity) return null;
  let order = 0;
  heapPush([h0, 0, order++, 0]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, negG, , s] = heapPop();
    const g = -negG;
    const boxes = boxesOf[s]!;
    if (bestG.get(stateKey(boxes, regionLowFrom(playerOf[s]!, boxes))) !== g) {
      // Stale entry: a cheaper route to this state was found already.
      // (regionLowFrom recomputes; see note below — we recompute the key
      // from the stored player square, which lies in the same region.)
      continue;
    }
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpansions) return null;
    expanded++;

    for (const b of boxes) occupied[b] = 1;
    stamp++;
    const here = stamp;
    // Mark the current reachable region.
    {
      let n = 0;
      stack[n++] = playerOf[s]!;
      seen[playerOf[s]!] = here;
      while (n > 0) {
        const p = stack[--n]!;
        for (const d of DIRS) {
          const q = p + d.dy * w + d.dx;
          if (!wall[q] && !occupied[q] && seen[q] !== here) {
            seen[q] = here;
            stack[n++] = q;
          }
        }
      }
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di]!;
        const step = d.dy * w + d.dx;
        const t = b + step;
        // The player must stand on the far side of the box to push it.
        if (seen[b - step] !== here) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(t);
        let key = '';
        let next: number[] | null = null;
        let low = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, bb) => a - bb);
          low = regionLow(b); // player stands on the box's old square
          key = stateKey(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const prevG = bestG.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        bestG.set(key, g + 1);
        const h = matchingHeuristic(next, goals, distRows);
        if (h === Infinity) continue;
        const ns = boxesOf.length;
        boxesOf.push(next);
        playerOf.push(b);
        gOf.push(g + 1);
        parentOf.push(s);
        pushOf.push({ from: b, to: t, dirIndex: di });
        heapPush([g + 1 + h, -(g + 1), order++, ns]);
        if (solved(next)) {
          found = ns;
          break;
        }
      }
      if (found >= 0) break;
    }
    for (const x of boxes) occupied[x] = 0;
    if (found >= 0) break;
  }
  if (found < 0) return null;

  const steps: PushStep[] = [];
  for (let s = found; s > 0; s = parentOf[s]!) steps.push(pushOf[s]!);
  steps.reverse();
  return steps;

  // Region's lowest square from a player square, given the boxes (used only
  // for the stale-entry key check above).
  function regionLowFrom(from: number, boxes: number[]): number {
    for (const b of boxes) occupied[b] = 1;
    const low = regionLow(from);
    for (const b of boxes) occupied[b] = 0;
    return low;
  }
}

// Turn a push sequence into moves: walk to each push square, then push.
function pushesToMoves(board: Board, steps: PushStep[]): string {
  const occupied = new Uint8Array(board.wall.length);
  let player = board.player;
  const boxes = board.boxes.slice();
  let moves = '';
  for (const step of steps) {
    const d = DIRS[step.dirIndex]!;
    const stepOff = d.dy * board.w + d.dx;
    for (const b of boxes) occupied[b] = 1;
    moves += shortestWalk(board, occupied, player, step.from - stepOff);
    for (const b of boxes) occupied[b] = 0;
    moves += d.push;
    player = step.from;
    boxes[boxes.indexOf(step.from)!] = step.to;
    boxes.sort((a, b) => a - b);
  }
  return moves;
}

// Split a level into rooms: cut the hallway row's dead squares (squares a
// box can never reach a goal from) out of the floor; each remaining part
// holding boxes is a room. Returns null when the level is no set of rooms.
function roomsOf(level: Board): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i]! + d.dy * w + d.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve one level room by room; fall back to a whole-level search when the
// level is no set of rooms. Expansion budgets are work, never the clock.
function solveLevel(level: Board): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const steps = solvePushes(level, 600_000);
    return steps === null ? '' : pushesToMoves(level, steps);
  }
  let player = level.player;
  let moves = '';
  const doneGoals: number[] = [];
  for (const room of rooms) {
    const wall = level.wall.slice();
    // Other rooms' boxes stand as walls; finished rooms contribute their
    // goals (their boxes sit on them).
    for (const other of rooms) {
      if (other === room) continue;
      const blockers = other.goals.every((g) => doneGoals.includes(g))
        ? other.goals
        : other.boxes;
      for (const b of blockers) wall[b] = 1;
    }
    const sub: Board = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const steps = solvePushes(sub, 600_000);
    if (steps === null) return '';
    const subMoves = pushesToMoves(sub, steps);
    moves += subMoves;
    // Advance the player: replay the sub-moves on the full board.
    for (const ch of subMoves.toLowerCase()) {
      player += ch === 'u' ? -level.w : ch === 'd' ? level.w : ch === 'l' ? -1 : 1;
    }
    doneGoals.push(...room.goals);
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  const moves = solveLevel(level);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
