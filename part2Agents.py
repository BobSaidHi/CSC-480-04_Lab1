from typing import overload

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
        # Get locations of interest
        wizardLocs = state.get_all_entity_locations(Wizard)
        if len(wizardLocs) == 0:
            return float("-inf")

        wizardLoc = wizardLocs[0]
        allCrystalLocs = state.get_all_entity_locations(Crystal)
        portalLoc = state.get_all_tile_locations(Portal)[0]
        allGoblinLocs = state.get_all_entity_locations(Goblin)

        # Higher is better
        score = 0.0

        # Constants
        CRYSTAL_SCORE = 50.0

        # Account for game scoring
        score += CRYSTAL_SCORE * state.score

        # Compute Manhattan distance to closest crystal (penalize high distance)
        if len(allCrystalLocs) > 0:
            minDistance = float("inf")
            for targetLoc in allCrystalLocs:
                a = wizardLoc.row - targetLoc.row
                b = wizardLoc.col - targetLoc.col
                distance = abs(a) + abs(b)
                if distance < minDistance:
                    minDistance = distance
            # if(minDistance <= 1):
            #     minDistance = 0
            crystalScore = CRYSTAL_SCORE / (minDistance + 1)
            score += crystalScore

        # Compute Manhattan distance to portal (penalize high distance)
        a = wizardLoc.row - portalLoc.row
        b = wizardLoc.col - portalLoc.col
        portalDistance = abs(a) + abs(b)
        portalScore = 35.0 / (portalDistance + 1)
        score += portalScore

        # Compute Manhattan distance to goblins (penalize low distance)
        goblinDistances = []

        for goblinLoc in allGoblinLocs:
            a = wizardLoc.row - goblinLoc.row
            b = wizardLoc.col - goblinLoc.col
            goblinDistance = abs(a) + abs(b)
            goblinDistances.append(goblinDistance)

        goblinDistances.sort()
        base = 95.0
        for distance in goblinDistances:
            if distance < 4:
                score -= base / (distance + 1)
                base *= 0.5

        return score

    def is_terminal(self, state: GameState) -> bool:
        # Get relevant locations
        wizardLocs = state.get_all_entity_locations(Wizard)

        # Check for no wizard
        if len(wizardLocs) == 0:
            return True

        # Get wizard
        wizard = wizardLocs[0]

        # Check if portal reached & out of crystals
        onPortal = isinstance(state.tile_grid[wizard.row][wizard.col], Portal)
        # noCrystals = len(state.get_all_entity_locations(Crystal)) == 0

        if onPortal: #and noCrystals:
            return True

        # Else: not done
        return False

    def react(self, state: GameState) -> WizardMoves:
        successorGameStates = self.get_successors(state)

        # Check if our of things to do
        if len(successorGameStates) == 0:
            return WizardMoves.STAY

        bestAction = WizardMoves.STAY
        bestScore = float("-inf")
        for possibleAction, possibleNextState in successorGameStates:
            possibleScore = self.minimax(possibleNextState, 0)
            if possibleScore > bestScore:
                bestScore = possibleScore
                bestAction = possibleAction

        return bestAction  # DONE - type

    def minimax(self, state: GameState, depth: int):
        # Base case - terminal
        if self.is_terminal(state):
            return self.evaluation(state)

        # Base case - max depth
        if depth >= self.max_depth:
            return self.evaluation(state)

        activeEntity = state.get_active_entity()
        successorGameStates = self.get_successors(state)

        # Check if terminal node in search
        if len(successorGameStates) == 0:
            return self.evaluation(state)

        # Wizard turn (MAXi node)
        if isinstance(activeEntity, Wizard):
            bestForWizard = float("-inf")
            for possibleAction, possibleState in successorGameStates:
                val = self.minimax(possibleState, depth + 1)
                if val > bestForWizard:
                    bestForWizard = val
            return bestForWizard

        # Worst: Goblin turn (MINi node)
        worstForWizard = float("inf")
        for possibleAction, possibleState in successorGameStates:
            val = self.minimax(possibleState, depth)  # depth unchanged
            if val < worstForWizard:
                worstForWizard = val
        return worstForWizard


class WizardAlphaBeta(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # Get locations of interest
        wizardLocs = state.get_all_entity_locations(Wizard)
        if len(wizardLocs) == 0:
            return float("-inf")

        wizardLoc = wizardLocs[0]
        allCrystalLocs = state.get_all_entity_locations(Crystal)
        portalLoc = state.get_all_tile_locations(Portal)[0]
        allGoblinLocs = state.get_all_entity_locations(Goblin)

        # Higher is better
        score = 0.0

        # Constants
        CRYSTAL_SCORE = 50.0

        # Account for game scoring
        score += CRYSTAL_SCORE * state.score

        # Compute Manhattan distance to closest crystal (penalize high distance)
        if len(allCrystalLocs) > 0:
            minDistance = float("inf")
            for targetLoc in allCrystalLocs:
                a = wizardLoc.row - targetLoc.row
                b = wizardLoc.col - targetLoc.col
                distance = abs(a) + abs(b)
                if distance < minDistance:
                    minDistance = distance
            # if(minDistance <= 1):
            #     minDistance = 0
            crystalScore = CRYSTAL_SCORE / (minDistance + 1)
            score += crystalScore

        # Compute Manhattan distance to portal (penalize high distance)
        a = wizardLoc.row - portalLoc.row
        b = wizardLoc.col - portalLoc.col
        portalDistance = abs(a) + abs(b)
        portalScore = 35.0 / (portalDistance + 1)
        score += portalScore

        # Compute Manhattan distance to goblins (penalize low distance)
        goblinDistances = []

        for goblinLoc in allGoblinLocs:
            a = wizardLoc.row - goblinLoc.row
            b = wizardLoc.col - goblinLoc.col
            goblinDistance = abs(a) + abs(b)
            goblinDistances.append(goblinDistance)

        goblinDistances.sort()
        base = 95.0
        for distance in goblinDistances:
            if distance < 4:
                score -= base / (distance + 1)
                base *= 0.5

        return score

    def is_terminal(self, state: GameState) -> bool:
        # Get relevant locations
        wizardLocs = state.get_all_entity_locations(Wizard)

        # Check for no wizard
        if len(wizardLocs) == 0:
            return True

        # Get wizard
        wizard = wizardLocs[0]

        # Check if portal reached & out of crystals
        onPortal = isinstance(state.tile_grid[wizard.row][wizard.col], Portal)
        # noCrystals = len(state.get_all_entity_locations(Crystal)) == 0

        if onPortal: # and noCrystals:
            return True

        # Else: not done
        return False

    def react(self, state: GameState) -> WizardMoves:
        successorGameStates = self.get_successors(state)

        # Check if our of things to do
        if len(successorGameStates) == 0:
            return WizardMoves.STAY

        bestAction = WizardMoves.STAY
        bestScore = float("-inf")
        alphaMaxed = float("-inf")
        betaMined = float("inf")

        # Evaluate successor states for optimal search order
        # orderedSuccessorGameStates = list(successorGameStates).sort()

        for possibleAction, possibleNextState in successorGameStates:
            possibleScore = self.alpha_beta_minimax(possibleNextState, 0)
            if possibleScore > bestScore:
                bestScore = possibleScore
                bestAction = possibleAction
            if bestScore > alphaMaxed:
                alphaMaxed = bestScore

        return bestAction  # DONE - type

    def alpha_beta_minimax(self, state: GameState, depth: int):
        return self.alpha_beta_minimax2(
            state, depth, float("-inf"), float("inf")
        )

    def alpha_beta_minimax2(
        self, state: GameState, depth: int, alphaMaxed, betaMined
    ):
        # Base case - terminal
        if self.is_terminal(state):
            return self.evaluation(state)

        # Base case - max depth
        if depth >= self.max_depth:
            return self.evaluation(state)

        activeEntity = state.get_active_entity()
        successorGameStates = self.get_successors(state)

        # Check if terminal node in search
        if len(successorGameStates) == 0:
            return self.evaluation(state)

        # Wizard turn (MAXi node)
        if isinstance(activeEntity, Wizard):
            bestForWizard = float("-inf")

            # Improve ordering
            # MAX node: best first (high to low)
            ordered = list(successorGameStates)
            ordered.sort(key=self.orderByScore, reverse=True)

            for possibleAction, possibleState in successorGameStates:
                val = self.alpha_beta_minimax2(
                    possibleState, depth + 1, alphaMaxed, betaMined
                )
                if val > bestForWizard:
                    bestForWizard = val
                if bestForWizard > alphaMaxed:
                    alphaMaxed = bestForWizard
                if alphaMaxed >= betaMined:
                    # Prune
                    break
            return bestForWizard

        # Else: Worst: Goblin turn (MINi node)
        worstForWizard = float("inf")

        # Improve ordering
        # MIN node: worst for wizard first (low to high)
        ordered = list(successorGameStates)
        ordered.sort(key=self.orderByScore)

        for possibleAction, possibleState in successorGameStates:
            val = self.alpha_beta_minimax2(
                possibleState, depth, alphaMaxed, betaMined
            )
            if val < worstForWizard:
                worstForWizard = val
            if worstForWizard < betaMined:
                betaMined = worstForWizard
            if alphaMaxed >= betaMined:
                # Prune
                break
        return worstForWizard

    def orderByScore(self, successor):
        action, nextState = successor
        return self.evaluation(nextState)


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
