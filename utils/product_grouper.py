from typing import List, Set, Tuple, Iterable

from tqdm import tqdm
import pandas as pd

class ProdcutGrouperBase:
    def insert(self, id1: int, id2: int, target: bool) -> None:
        raise NotImplementedError

    def get_same_product_sets(self) -> Iterable[Set[int]]:
        raise NotImplementedError
    
    def get_different_product_sets(self) -> Iterable[Set[int]]:
        raise NotImplementedError


class ProductGrouper_list(ProdcutGrouperBase):
    def __init__(self):
        self.same_products_l_of_s: List[Set[int]] = []
        self.different_products_l_of_s: List[Set[int]]  = []
    
    def insert(self, id1: int, id2: int, target: bool) -> None:
        list_of_sets = self.same_products_l_of_s if target else self.different_products_l_of_s
        
        i1 = self._find(id1, list_of_sets)
        i2 = self._find(id2, list_of_sets)
        if i1 != -1 and id2 != -1:
            list_of_sets[i1].update(list_of_sets[i2])
            del list_of_sets[i2]
        elif i1 == -1 and id2 == -1:
            list_of_sets.append({id1, id2})
        elif i1 == -1 and i2 != -1:
            list_of_sets[i1].add(id2)
        else: # i1 != -1 and i2 == -1
            list_of_sets[i2].add(id1)
        
    @staticmethod
    def _find(product_id: int, list_of_sets: List[Set[int]]) -> int:
        for i, s in enumerate(list_of_sets):
            if product_id in s:
                return i
        return -1
    
    def get_same_product_sets(self) -> Iterable[Set[int]]:
        return self.same_products_l_of_s
    
    def get_different_product_sets(self) -> Iterable[Set[int]]:
        return self.different_products_l_of_s

    
class ProductGrouper_dict(ProdcutGrouperBase):
    def __init__(self):
        self.id_to_set_same: dict[int, Set[int]] = {}
        self.id_to_set_different: dict[int, Set[int]] = {}
    
    def insert(self, id1: int, id2: int, target: bool) -> None:
        id_to_set = self.id_to_set_same if target else self.id_to_set_different
        
        has_id1 = id1 in id_to_set
        has_id2 = id2 in id_to_set
        if has_id1 and has_id2:
            id_to_set[id1].update(id_to_set[id2])
            id_to_set[id2] = id_to_set[id1]
        elif has_id1 and not has_id2:
            id_to_set[id1].add(id2)
            id_to_set[id2] = id_to_set[id1]
        elif not has_id1 and has_id2:
            id_to_set[id2].add(id1)
            id_to_set[id1] = id_to_set[id2]
        else: # not has_id1 and not has_id2:
            id_to_set[id1] = id_to_set[id2] = {id1, id2}
    
    def get_same_product_sets(self) -> Iterable[Set[int]]:
        return set(frozenset(v) for v in self.id_to_set_same.values())
    
    def get_different_product_sets(self) -> Iterable[Set[int]]:
        return set(frozenset(v) for v in self.id_to_set_different.values())


def get_product_groups(dataset: pd.DataFrame) -> Tuple[Iterable[Set[int]], Iterable[Set[int]]]:
    ProductGrouper = ProductGrouper_dict
    product_groups = ProductGrouper()
    for _, (target, id1, id2) in tqdm(dataset.iterrows(), total = len(dataset)):
        product_groups.insert(id1, id2, target)

    same_prods_sets = product_groups.get_same_product_sets()
    different_prods_sets = product_groups.get_different_product_sets()

    return same_prods_sets, different_prods_sets