use std::{collections::{BTreeMap, BTreeSet, HashMap, HashSet}, env::current_exe, fmt, path::Path, thread::current};

use anyhow::bail;
use itertools::Itertools;

/// For bitmasks, we will use the convention that 
/// the positions have the following locations
/// 
/// 0 1 2
/// 7 X 3
/// 6 5 4

enum Direction {
    Up, Down, Left, Right
}

impl Direction {
    // A bitmask that will fill the 3 bits
    // that correspond to the direction
    pub fn bitmask(&self) -> BitMask {
        match self {
            Self::Up => 0b11100000,
            Self::Down => 0b00001110,
            Self::Left => 0b10000011,
            Self::Right => 0b00111000,
        }
    }
}

fn bitmask_offsets() -> impl Iterator<Item = (isize, isize)> {
    [
        (-1, 1), (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1), (-1, -1), (-1, 0)
    ].into_iter().rev()
}

// Represents the value of a single cell in the 
// tapa board. If the option is None, we don't know
// what the cell is
type Cell = Option<bool>;
type BitMask = u8;

/// There are only a certain number of different clue types,
/// which we list here
#[derive(Clone, Copy, Debug)]
pub enum Clue {
    // Single digit clues
    S0, S1, S2, S3, S4, S5, S6, S7,

    // Two digit clues
    D11, D12, D13, D14, D15, D22, D23, D24, D33,

    // Three digit clues
    T111, T112, T113, T122,

    // Four digit clues
    Q1111
}

impl Clue {
    pub fn from_digits(digits: impl IntoIterator<Item=usize>) -> anyhow::Result<Self> {
        let mut digits = digits.into_iter().collect_vec();
        if digits.len() > 4 { bail!("No clue can be more than 4 digits") }
        digits.sort_unstable();

        Ok(match digits[..] {
            [0] => Self::S0,
            [1] => Self::S1,
            [2] => Self::S2,
            [3] => Self::S3,
            [4] => Self::S4,
            [5] => Self::S5,
            [6] => Self::S6,
            [7] => Self::S7,

            [1, 1] => Self::D11,
            [1, 2] => Self::D12,
            [1, 3] => Self::D13,
            [1, 4] => Self::D14,
            [1, 5] => Self::D15,
            [2, 2] => Self::D22,
            [2, 3] => Self::D23,
            [2, 4] => Self::D24,
            [3, 3] => Self::D33,

            [1, 1, 1] => Self::T111,
            [1, 1, 2] => Self::T112,
            [1, 1, 3] => Self::T113,
            [1, 2, 2] => Self::T122,

            [1, 1, 1, 1] => Self::Q1111,

            _ => bail!("Invalid clue")
        })
    }
    
    pub fn bitmasks(&self) -> impl Iterator<Item = BitMask> {
        match self {
            Self::S0 => vec![0b00000000],
            Self::S1 => vec![0b00000001],
            Self::S2 => vec![0b00000011],
            Self::S3 => vec![0b00000111],
            Self::S4 => vec![0b00001111],
            Self::S5 => vec![0b00011111],
            Self::S6 => vec![0b00111111],
            Self::S7 => vec![0b01111111],

            Self::D11 => vec![0b00000101, 0b00001001, 0b00010001],
            Self::D12 => vec![0b00001101, 0b00011001],
            Self::D13 => vec![0b00011101, 0b00111001],
            Self::D14 => vec![0b00111101],
            Self::D15 => vec![0b01111101],
            Self::D22 => vec![0b00011011, 0b00110011],
            Self::D23 => vec![0b00111011],
            Self::D24 => vec![0b01111011],
            Self::D33 => vec![0b01110111],

            Self::T111 => vec![0b00010101, 0b00100101],
            Self::T112 => vec![0b00110101, 0b01101001],
            Self::T113 => vec![0b01110101],
            Self::T122 => vec![0b01101101],

            Self::Q1111 => vec![0b01010101],
        }.into_iter().flat_map(|bitvector: u8| (0..8).map(move |rotate| bitvector.rotate_right(rotate)))
            .flat_map(|bitmask| [bitmask, bitmask.reverse_bits()])
            .unique()
    }

    fn to_digits(&self) -> Vec<usize> {
        match self {
            Self::S0 => vec![0],
            Self::S1 => vec![1],
            Self::S2 => vec![2],
            Self::S3 => vec![3],
            Self::S4 => vec![4],
            Self::S5 => vec![5],
            Self::S6 => vec![6],
            Self::S7 => vec![7],
            Self::D11 => vec![1, 1],
            Self::D12 => vec![1, 2],
            Self::D13 => vec![1, 3],
            Self::D14 => vec![1, 4],
            Self::D15 => vec![1, 5],
            Self::D22 => vec![2, 2],
            Self::D23 => vec![2, 3],
            Self::D24 => vec![2, 4],
            Self::D33 => vec![3, 3],
            Self::T111 => vec![1, 1, 1],
            Self::T112 => vec![1, 1, 2],
            Self::T113 => vec![1, 1, 3],
            Self::T122 => vec![1, 2, 2],
            Self::Q1111 => vec![1, 1, 1, 1],
        }
    }
}

fn bitmask_filter(pattern: BitMask) -> impl Fn(&BitMask) -> bool {
    move |bitmask| {
        *bitmask & pattern == 0
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    cells: Vec<Cell>,
    height: usize,
    width: usize,

    clues: BTreeMap<(usize, usize), Clue>,
}

enum SolveResult {
    Failure, 
    Unknown,
    Solved(Board)
}

impl Board {
    pub fn new(height: usize, width: usize, clues: BTreeMap<(usize, usize), Clue>) -> anyhow::Result<Self> {
        if height == 0 || width == 0 { bail!("Cannot have a zero in the dimension") }

        if clues.keys().any(|(x, y)| *x >= width || *y >= width) {
            bail!("Clue out of bounds")
        }

        let mut board = Self {
            cells: std::iter::repeat(None).take(height * width).collect_vec(),
            height, width, clues
        };

        for (x, y) in board.clues.keys() {
            board.cells[y * width + *x] = Some(false)
        }

        Ok(board)
    }

    pub fn set_cell(&mut self, (x, y): (usize, usize), value: bool) {
        assert!(self.is_inside((x, y)), "Position outside board");
        assert!(self.get_cell((x, y)).is_none(), "Cell ({x}, {y}) already set on\n{self}");

        self.cells[y * self.width + x] = Some(value);
    }

    pub fn get_cell(&self, (x, y): (usize, usize)) -> Cell {
        assert!(self.is_inside((x, y)), "Position outside board");

        self.cells[y * self.width + x]
    }

    pub fn get_clue(&self, c: (usize, usize)) -> Option<Clue> {
        self.clues.get(&c).copied()
    }

    pub fn is_inside(&self, (x, y): (usize, usize)) -> bool {
        x < self.width && y < self.height
    }

    pub fn is_inside_isize(&self, (x, y): (isize, isize)) -> bool {
        x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height
    }

    // Iterator that gives bitpos and the coordinates of surrounding cells. This filters off out of
    // bounds positions
    fn surrounds(&self, (x, y): (usize, usize)) -> impl Iterator<Item = (usize, (usize, usize))> + '_{
        bitmask_offsets().enumerate()
            .filter(move |(_, (x_offset, y_offset))| *x_offset >= -(x as isize) || *y_offset >= -(y as isize))
            .map(move |(bitpos, (x_offset, y_offset))| (bitpos, ((x as isize + x_offset) as usize, (y as isize + y_offset) as usize)))
            .filter(move |(_, pos)| self.is_inside(*pos))
    }

    // Gives the surrounding cells that are directly adjacent to the cell. Not on
    // the diagonals
    fn surrounds_adjacent(&self, (x, y): (usize, usize)) -> impl Iterator<Item = (usize, (usize, usize))> + '_{
        bitmask_offsets().enumerate()
            // We filter diagonals by the fact that they're 2 in manhattan distance away
            .filter(|(_, (x_offset, y_offset))| (x_offset.abs() + y_offset.abs()) % 2 == 1)
            .filter(move |(_, (x_offset, y_offset))| *x_offset >= -(x as isize) || *y_offset >= -(y as isize))
            .map(move |(bitpos, (x_offset, y_offset))| (bitpos, ((x as isize + x_offset) as usize, (y as isize + y_offset) as usize)))
            .filter(move |(_, pos)| self.is_inside(*pos))
    }

    fn unknown_cells_around(&self, (x, y): (usize, usize)) -> BitMask {
        self.surrounds((x, y))
            .fold(0, |bitmask, (bitpos, coord)| bitmask | ((self.get_cell(coord).is_none() as BitMask) << bitpos))
    }

    fn set_cells_around(&self, (x, y): (usize, usize)) -> BitMask {
        self.surrounds((x, y))
            .fold(0, |bitmask, (bitpos, coord)| bitmask | ((self.get_cell(coord).unwrap_or(false) as BitMask) << bitpos))
    }

    fn all_coordinates(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.width).flat_map(|x| (0..self.width).map(move |y| (x, y)))
    }

    pub fn solve_clue_surroundings(&mut self) -> bool {
        let mut optimized = false;
        let clues = self.clues.iter().map(|(pos, clue)| (*pos, *clue)).collect_vec();

        // If we have invalidated clues, short circuit and return false.
        // The algorithm below requires there to be at least one valid bitmask
        // per clue, and this validates that there is at least one
        if !self.is_solved_valid_clues() {
            return false
        }

        for (pos, clue) in clues {
            // println!("Clue: {:?}", clue);
            // println!("Surrounds: {:?}", self.surrounds(pos).map(|(_, pos)| (pos, self.get_cell(pos))).collect_vec());

            let unknown_mask = self.unknown_cells_around(pos);
            let set_cells_around = self.set_cells_around(pos);
            // println!("For clue ({:?}, {:?}), we have unknown mask and set mask: {:#010b}, {:#010b}", pos, clue.to_digits(), unknown_mask, set_cells_around);

            let (new_set_bitmask, new_unset_bitmask) = clue.bitmasks()
                .filter(|bitmask| {
                    let known_cells = bitmask & !unknown_mask;
                    let valid = (known_cells & bitmask) | (known_cells & !bitmask) == set_cells_around;
                    // println!("Known cells: {known_cells:#010b}, Bitmask {bitmask:#010b} valid: {valid}");
                    valid
                })
                // .map(|v| { println!("Valid bitmask {v:#010b}"); v })
                .fold((unknown_mask, unknown_mask), |(common_set, common_unset), bitmask| (common_set & bitmask, common_unset & !bitmask));
            // println!("New set bitmask and unset bitmask: {:#010b}, {:#010b}", new_set_bitmask, new_unset_bitmask);

            let set_positions = self.surrounds(pos)
                .filter(|(bitpos, _)| new_set_bitmask & (1 << bitpos) != 0)
                .map(|(_, pos)| pos)
                .collect_vec();

            for position in set_positions {
                self.set_cell(position, true);
                optimized = true;
            }

            let unset_positions = self.surrounds(pos)
                .filter(|(bitpos, _)| new_unset_bitmask & (1 << bitpos) != 0)
                .map(|(_, pos)| pos)
                .collect_vec();

            for position in unset_positions {
                self.set_cell(position, false);
                optimized = true;
            }

            if optimized == true {
                break
            }
        }

        optimized
    }

    // If we find a corner, a 2x2 piece with 3 of the cells shaded in
    // and one unknown, the unknown is a false
    fn solve_invalidate_corners(&mut self) -> bool {
        if !self.is_solved_no_blocks() {
            return false;
        }

        let unset_coordinates = self.all_coordinates().filter(|(x, y)| (*x + 1 != self.width) && (*y + 1 != self.height))
            .filter_map(|(x, y)| {
                let (a, b, c, d) = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1));
          
                match (self.get_cell(a), self.get_cell(b), self.get_cell(c), self.get_cell(d)) {
                    (Some(true), Some(true), Some(true), None) => Some(d),
                    (Some(true), Some(true), None, Some(true)) => Some(c),
                    (Some(true), None, Some(true), Some(true)) => Some(b),
                    (None, Some(true), Some(true), Some(true)) => Some(a),
                    _ => None
                }
            }).collect::<HashSet<_>>();

        let optimized = unset_coordinates.len() > 0;
        for unset_pos in unset_coordinates {
            self.set_cell(unset_pos, false);
        }

        optimized
    }

    // Given a cell, it will return two sets. All filled cells that are connected to that
    // cell through other filled cells, and a set that represents all unknown cells that are 
    // adjacent to cells in the first set
    //
    // This should only be called on filled cells
    fn flood_fill_from(&self, cell: (usize, usize)) -> (HashSet<(usize, usize)>, HashSet<(usize, usize)>) {
        assert_eq!(self.get_cell(cell), Some(true));

        let mut filled = HashSet::new();
        let mut adjacent_unknowns = HashSet::new();

        let mut queue = Vec::from([ cell ]);
        while let Some(cell) = queue.pop() {
            if filled.contains(&cell) || adjacent_unknowns.contains(&cell) { continue; } 

            match self.get_cell(cell) {
                Some(true) => {
                    filled.insert(cell);
                    for (_, cell) in self.surrounds_adjacent(cell) {
                        queue.push(cell);
                    }
                },
                Some(false) => continue,
                None => { adjacent_unknowns.insert(cell); }
            }

        }

        (filled, adjacent_unknowns)
    }

    fn solve_flood_fill(&mut self) -> bool {
        let mut scanned = HashSet::new();
        let mut found_cells = HashSet::new();

        for cell in self.all_coordinates() {
            if self.get_cell(cell) != Some(true) || scanned.contains(&cell) { continue; }            

            let (filled, adjacent_unknowns) = self.flood_fill_from(cell);
            scanned.extend(filled.into_iter());

            if adjacent_unknowns.len() == 1 {
                let found_cell = adjacent_unknowns.into_iter().next().unwrap();
                found_cells.insert(found_cell);
            }
        }

        for cell in found_cells.iter() {
            self.set_cell(*cell, true);
        }

        found_cells.len() > 0
    }

    fn contains_islands(&self) -> bool {
        // Check if we fill in all the unknowns, if it's all one connected component
        let mut board = self.clone();
        let unknowns = board.all_coordinates().filter(|coordinate| board.get_cell(*coordinate).is_none())
            .collect_vec();

        for unknown in unknowns {
            board.set_cell(unknown, true);
        }

        let filled_cells = board.all_coordinates().filter(|coordinate| board.get_cell(*coordinate) == Some(true))
            .collect::<HashSet<_>>();
        if filled_cells.len() < 1 { return false; }
        let some_cell = filled_cells.iter().next().unwrap();
        let (island, _) = board.flood_fill_from(*some_cell);

        return filled_cells != island;
    }

    // Rudimentary checks to gague whether this board is solvable or not.
    // Will return true if the board is definitely not solvable, and false
    // otherwise
    fn is_board_unsolvable(&self) -> bool {
        self.contains_islands()
    }

    fn is_possible_bitmask(&self, position: (usize, usize), bitmask: BitMask) -> bool {
        let mut surrounds_bits = 0;

        for (bitpos, cell) in self.surrounds(position) {
            surrounds_bits |= (1 << bitpos);
            match self.get_cell(cell) {
                None => continue,
                Some(true) => {
                    if bitmask & (1 << bitpos) == 0 { return false; }
                }
                Some(false) => {
                    if bitmask & (1 << bitpos) != 0 { return false; }
                }
            }
        }

        bitmask & !surrounds_bits == 0
    }

    // 
    fn solve_guess_position(&mut self, position: (usize, usize), current_depth: usize, max_depth: usize) -> SolveResult {
        // println!("Guessing on ({}, {}) with depth {current_depth} on\n{self}", position.0, position.1); 
        if current_depth == 0 {
            let mut board = self.clone();
            board.set_cell(position, false);
            match board.solve(current_depth + 1, current_depth + 1) {
                SolveResult::Failure => {
                    self.set_cell(position, true);
                    return self.solve(current_depth, max_depth)
                },
                SolveResult::Solved(board) => return SolveResult::Solved(board),
                SolveResult::Unknown => ()
            };

            let mut board = self.clone();
            board.set_cell(position, true);
            match board.solve(current_depth + 1, current_depth + 1) {
                SolveResult::Failure => {
                    self.set_cell(position, false);
                    return self.solve(current_depth, max_depth)
                },
                SolveResult::Solved(board) => return SolveResult::Solved(board),
                SolveResult::Unknown => ()
            };
        }

        let mut board = self.clone();
        board.set_cell(position, false);
        let unset_result = board.solve(current_depth + 1, max_depth);
        if let SolveResult::Solved(_) = unset_result { return unset_result; }

        let mut board = self.clone();
        board.set_cell(position, true);
        let set_result = board.solve(current_depth + 1, max_depth);
        if let SolveResult::Solved(_) = set_result { return unset_result; }

        match (unset_result, set_result) {
            (SolveResult::Failure, SolveResult::Failure) => SolveResult::Failure,
            _ => SolveResult::Unknown
        }
    }

    // This is a heuristic. We want to guess at coordinates that will hopefully be faster
    // at either being proven wrong, or that are right. Therefore, we will prioritize coordinates that
    // are around clues than coordinates that are not around clues already.
    fn guess_coordinate_order(&self) -> Vec<(usize, usize)> {
        // A mapping of coordinate to number of possible bitmaps 
        let mut clue_adjacent_positions = self.clues.iter().flat_map(|(position, clue)| {
            let count = clue.bitmasks().filter(|bitmask| self.is_possible_bitmask(*position, *bitmask))
                .count();
            self.surrounds(*position)
                .filter(|(_, surrounding)| self.get_cell(*surrounding).is_none())
                .map(move |(_, surrounds)| (surrounds, count))
        })
            .sorted_by_cached_key(|(_, count)| *count)
            .map(|(position, _count)| position)
            .unique()
            .collect_vec();

        let adjacent_positions_set = clue_adjacent_positions.clone().into_iter().collect::<HashSet<_>>();
        let non_adjacent_positions = self.all_coordinates().filter(|coordinate| self.get_cell(*coordinate).is_none())
            .filter(|coordinate| !adjacent_positions_set.contains(coordinate));

        clue_adjacent_positions.extend(non_adjacent_positions);
        clue_adjacent_positions
            

        // let clue_adjacents = self.clues.keys()
        //     .flat_map(|coordinate| self.surrounds(*coordinate).map(|(_, v)| v))
        //     .collect::<HashSet<_>>();

        // self.all_coordinates()
        //     .sorted_by_key(|coordinate| if clue_adjacents.contains(coordinate) { 1 } else { 0 })
        //     .collect_vec()
    }

    fn solve_guess(&mut self, current_depth: usize, max_depth: usize) -> SolveResult {
        let mut unknown_guess = false;
        let coordinates = self.guess_coordinate_order();

        for position in coordinates {
            if self.get_cell(position).is_none() {
                match self.solve_guess_position(position, current_depth, max_depth) {
                    SolveResult::Solved(board) => return SolveResult::Solved(board),
                    SolveResult::Unknown => unknown_guess = true,
                    SolveResult::Failure => ()
                };
            }
        }

        match unknown_guess {
            true => SolveResult::Unknown,
            false => SolveResult::Failure
        }
    }

    // Tries to solve the board, knowing that we're at the 'current_depth' in
    // terms of guesses, given a max depth amount of guesses
    fn solve(&mut self, current_depth: usize, max_depth: usize) -> SolveResult {
        if current_depth > max_depth { return SolveResult::Unknown; }
        if current_depth < 3 {
            // println!("Running 'solve({current_depth}, {max_depth})' on:\n{self}");
        }

        while self.solve_clue_surroundings() || self.solve_invalidate_corners() || self.solve_flood_fill() {}
        // println!("After trivial solving:\n{self}");
        // println!("Checking if solved");
        match self.is_solved() {
            value @ (SolveResult::Failure | SolveResult::Solved(_)) => return value,
            SolveResult::Unknown => ()
        };
        if self.is_board_unsolvable() { return SolveResult::Failure; }

        // println!("Got:\n{self}");

        if current_depth == max_depth { 
            SolveResult::Unknown 
        } else {
            self.solve_guess(current_depth, max_depth)
        }

    }

    fn is_solved_no_unknowns(&self) -> bool {
        self.all_coordinates().filter(|coordinate| self.get_cell(*coordinate).is_none())
            .count() == 0
    }

    fn is_solved_all_connected(&self) -> bool {
        let all_filled = self.all_coordinates().filter(|coordinate| self.get_cell(*coordinate) == Some(true))
            .collect::<HashSet<_>>();

        if all_filled.len() == 0 { return false; }
        let some_cell = *all_filled.iter().next().unwrap();
        
        let (filled, adjacent_unknown) = self.flood_fill_from(some_cell);
        filled == all_filled && adjacent_unknown.len() == 0
    }

    fn is_solved_valid_clues(&self) -> bool {
        for (position, clue) in &self.clues {
            if !clue.bitmasks().any(|bitmask| self.is_possible_bitmask(*position, bitmask)) {
                return false
            }
        }

        return true
    }

    fn is_solved_no_blocks(&self) -> bool {
        self.all_coordinates().filter(|(x, y)| (*x + 1 != self.width) && (*y + 1 != self.height))
            .filter(|(x, y)| {
                self.get_cell((*x, *y)) == Some(true) &&
                self.get_cell((x + 1, *y)) == Some(true) &&
                self.get_cell((*x, y + 1)) == Some(true) &&
                self.get_cell((x + 1, y + 1)) == Some(true)
            })
            .count() == 0
    }

    // Conditions for being solved: No 2x2 area of filled. No unknowns,
    // and all cells must be connected
    fn is_solved(&self) -> SolveResult {
        if !self.is_solved_no_blocks() {
            return SolveResult::Failure
        }

        if !self.is_solved_valid_clues() {
            return SolveResult::Failure
        }

        if !self.is_solved_no_unknowns() {
            return SolveResult::Unknown
        }

        if !self.is_solved_all_connected() {
            return SolveResult::Failure
        }

        SolveResult::Solved(self.clone())
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut format_str = String::new();

        let full_square = "██".to_owned();
        let empty_square = "  ".to_owned();
        let unknown_square = "░░".to_owned();

        for y in (0..self.height).rev() {
            for level_y in 0..2 {
                for x in 0..self.width {

                    for level_x in 0..2 {
                        let cell = self.get_cell((x, y));
                        let clue = self.get_clue((x, y)).and_then(|clue| clue.to_digits().get(level_y * 2 + level_x).copied());

                        match (cell, clue) {
                            (Some(true), _) => format_str += &full_square,
                            (None, _) => format_str += &unknown_square,
                            (Some(false), None) => format_str += &empty_square,
                            (Some(false), Some(digit)) => format_str += &format!(" {digit}"),
                            _ => unreachable!()
                        }
                    }
                }

                format_str += "\n";
            }
        }

        write!(f, "{format_str}")
    } 
}

fn main() -> anyhow::Result<()> {
    // let mut board = Board::new(6, 6, BTreeMap::from([
    //     ((1, 1), Clue::from_digits([1, 4]).unwrap()),
    //     ((1, 2), Clue::from_digits([1, 3]).unwrap()),
    //     ((3, 1), Clue::from_digits([2, 4]).unwrap()),
    //     ((4, 1), Clue::from_digits([7]).unwrap()),
    //     ((1, 4), Clue::from_digits([6]).unwrap()),
    //     ((2, 4), Clue::from_digits([1, 1, 3]).unwrap()),
    //     ((4, 3), Clue::from_digits([6]).unwrap()),
    //     ((4, 4), Clue::from_digits([1, 5]).unwrap()),
    // ])).unwrap();
    // let mut board = Board::new(6, 6, BTreeMap::from([
    //     ((0, 0), Clue::from_digits([2]).unwrap()),
    //     ((5, 0), Clue::from_digits([2]).unwrap()),
    //     ((0, 5), Clue::from_digits([2]).unwrap()),
    //     ((5, 5), Clue::from_digits([3]).unwrap()),
    //     ((2, 1), Clue::from_digits([1, 1, 2]).unwrap()),
    //     ((3, 1), Clue::from_digits([6]).unwrap()),
    //     ((2, 4), Clue::from_digits([1, 5]).unwrap()),
    //     ((3, 4), Clue::from_digits([2, 3]).unwrap()),
    // ])).unwrap();

    // println!("Clues: {:?}", load_clues_from_file("puzzle.json", 20, 20).unwrap());
    let clues = load_clues_from_file("puzzle_easy.json", 20, 20)?;
    let mut board = Board::new(20, 20, BTreeMap::from(clues))?;
    println!("Board: {}", board);

    // let mut board = Board::new(6, 6, BTreeMap::from([
    //     ((1, 1), Clue::from_digits([6]).unwrap()),
    //     ((0, 2), Clue::from_digits([3]).unwrap()),
    //     ((1, 4), Clue::from_digits([7]).unwrap()),
    //     ((4, 1), Clue::from_digits([1, 3]).unwrap()),
    //     ((4, 4), Clue::from_digits([3, 3]).unwrap()),
    //     ((5, 2), Clue::from_digits([2]).unwrap()),
    // ])).unwrap();

    let mut max_depth = 1;
    loop {
        match board.solve(0, max_depth) {
            SolveResult::Solved(board) => {
               println!("Solved:\n{}", board);
               break;
            }
            SolveResult::Unknown => {
                max_depth += 1;
                println!("Extending depth to {}!", max_depth);
                continue;
            }
            SolveResult::Failure => {
                println!("Unsolvable board");
                break;
            }
        };
    }


    Ok(())
}

fn load_clues_from_file<P: AsRef<Path>>(path: P, width: usize, height: usize) -> anyhow::Result<BTreeMap<(usize, usize), Clue>> {
    let clue_list: Vec<Option<Vec<usize>>> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut clues = BTreeMap::new();

    for (position, clue) in clue_list.into_iter().enumerate().filter_map(|(pos, clue)| clue.map(|v| (pos, v))) {
        let (x, y) = (position % width, height - position / height - 1);
        clues.insert((x, y), Clue::from_digits(clue)?);
    }

    Ok(clues)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_edge_filter() {
        let clue = Clue::from_digits([1, 3]).unwrap();

        assert_eq!(
            clue.bitmasks().filter(bitmask_filter(Direction::Up.bitmask())).collect::<HashSet<BitMask>>(), 
            HashSet::from([0b00010111, 0b00011101])
        );
    }
}