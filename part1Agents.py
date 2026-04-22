from model import (
    Location,
    Portal,
    EmptyEntity,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import WizardSearchAgent
import heapq
from dataclasses import dataclass


# @brief "agent class that performs Depth First Search"
#
# @details In order to perform a search a `WizardSearchAgent`
# has the ability to choose game states to expand and learn
# their successors (this is the search transition function)
# as well as must have a way to process each new state that
# is found through a node expansion.
class WizardDFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    ##
    # @brief DFS stack / frontier of search states to expand
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row,
                search_state.wizard_loc.col,
                initial_wizard,
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    ##
    # @brief Check if the wizard location matches the portal location
    # @returns True if the goal (portal) has been reached, False otherwise
    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    # @brief "choose game states to expand"
    # @details Will be called repeatedly while a `plan` is not available yet
    # (empty), unless no next search expansion is returned
    # @details Pops a search state from DFS frontier LIFO stack.
    # Use search_to_game() as needed and return the node to expand.
    # @returns the next game state to expand, or nothing if there are
    # no more states to expand
    def next_search_expansion(self) -> GameState | None:
        # Check frontier
        # @details: In Python, len is a separate function, not a class method
        if(len(self.search_stack) == 0):
            return None
        
        # Else: Pop a search state from DFS frontier LIFO stack
        nextSearchState = self.search_stack.pop()

        # Check if goal has been reached
        if(self.is_goal(nextSearchState)):
            self.plan = list(reversed(self.paths[nextSearchState]))
            return None
        
        # Else: Return the state / node to expand next
        return self.search_to_game(nextSearchState)

    # @brief Process a successor node
    # @param source: one of the successors (an expanded game state . node) of the returned node from
    # @param target
    # @param WizardMoves action that transitions source -> target.
    # @details converts GameState to SearchState and checks against the list of
    # already visited states, update path, frontier, and `self.plan` upon
    # reaching the goal
    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        sourceSearchState = self.game_to_search(source)
        targetSearchState = self.game_to_search(target)

        # Check if the target search state has already been visited (in paths)
        if targetSearchState in self.paths:
            return
        
        # Else: Build new path
        # @details:  Copy the actions / path dict and append the new action
        newPath = self.paths[sourceSearchState] + [action]

        # Save the path
        self.paths[targetSearchState] = newPath

        # Check if goal has been reached
        if self.is_goal(targetSearchState):
            self.plan = list(reversed(newPath))
            return
        
        # Else: Add to frontier (push)
        self.search_stack.append(targetSearchState)

class WizardBFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row,
                search_state.wizard_loc.col,
                initial_wizard,
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        # TODO: YOUR CODE HERE
        raise NotImplementedError

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO: YOUR CODE HERE
        raise NotImplementedError


class WizardAstar(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row,
                search_state.wizard_loc.col,
                initial_wizard,
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def cost(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> float:
        return 1

    def heuristic(self, target: GameState) -> float:
        # TODO: YOUR CODE HERE
        raise NotImplementedError

    def next_search_expansion(self) -> GameState | None:
        # TODO: YOUR CODE HERE
        raise NotImplementedError

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO: YOUR CODE HERE
        raise NotImplementedError


class CrystalSearchWizard(WizardSearchAgent):
    # TODO: YOUR CODE HERE

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def next_search_expansion(self) -> GameState | None:
        # TODO YOUR CODE HEREs
        raise NotImplementedError

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO YOUR CODE HERE
        raise NotImplementedError


class SuboptimalCrystalSearchWizard(CrystalSearchWizard):
    def heuristic(self, target: SearchState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError
