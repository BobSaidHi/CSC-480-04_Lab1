from model import (
    Location,
    Portal,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import ReasoningWizard
from dataclasses import dataclass


class WizardGreedy(ReasoningWizard):
    def evaluation(self, state: GameState) -> float:
        # Get locations of interest
        wizardLoc = state.get_all_entity_locations(Wizard)[0]
        allCrystalLocs = state.get_all_entity_locations(Crystal)
        portalLoc = state.get_all_tile_locations(Portal)[0]
        allGoblinLocs = state.get_all_entity_locations(Goblin)

        # Higher is better
        score = 0.0

        # Compute Manhattan distance to closest crystal (penalize high distance)
        if len(allCrystalLocs) > 0:
            minDistance = float("inf")
            for targetLoc in allCrystalLocs:
                a = wizardLoc.row - targetLoc.row
                b = wizardLoc.col - targetLoc.col
                distance = abs(a) + abs(b)
                if distance < minDistance:
                    minDistance = distance
            crystalScore = 50.0 / (minDistance + 1)
            score += crystalScore

        # Compute Manhattan distance to portal (penalize high distance)
        a = wizardLoc.row - portalLoc.row
        b = wizardLoc.col - portalLoc.col
        portalDistance = abs(a) + abs(b)
        portalScore = 20.0 / (portalDistance + 1)
        score += portalScore

        # Compute Manhattan distance to goblins (penalize low distance)
        goblinDistances = []

        for goblinLoc in allGoblinLocs:
            a = wizardLoc.row - goblinLoc.row
            b = wizardLoc.col - goblinLoc.col
            goblinDistance = abs(a) + abs(b)
            goblinDistances.append(goblinDistance)

        goblinDistances.sort()
        base = 100.0
        for distance in goblinDistances:
            score -= base / (distance + 1)
            base *= 0.5

        return score

    # def is_terminal(self, state: GameState) -> bool:
    #     # TODO YOUR CODE HERE
    #     raise NotImplementedError


class WizardMiniMax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def minimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError


class WizardAlphaBeta(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def alpha_beta_minimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError


class WizardExpectimax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def expectimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError
