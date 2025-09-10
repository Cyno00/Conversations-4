from models.player import Item, Player, PlayerSnapshot


class Player3(Player):
    def __init__(self, snapshot: PlayerSnapshot, conversation_length: int) -> None:
        super().__init__(snapshot, conversation_length)

    def propose_item(self, history: list[Item]) -> Item | None:
        # filter out items already contributed
        candidates = [item for item in self.memory_bank if item not in self.contributed_items]

        if not candidates:
            return None

        # choose the item with max importance
        best_item = max(candidates, key=lambda i: i.importance)

        # track it so we don’t reuse it later
        self.contributed_items.append(best_item)

        return best_item
