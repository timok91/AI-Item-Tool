# item_generation/item_validator.py

from typing import Dict, List, Union
import re
import streamlit as st

class GermanItemValidator:
    """
    Validates German personality test items based on psychometric best practices.
    """
    
    def __init__(self):
        self.extreme_words = {
            'immer', 'nie', 'alle', 'keiner', 'absolut', 'völlig', 
            'komplett', 'total', 'äußerst', 'extrem', 'ausnahmslos',
            'definitiv', 'grundsätzlich', 'zweifellos', 'unmöglich',
            'ausschließlich', 'durchgehend', 'gänzlich', 'vollständig'
        }
        
        self.desirability_markers = {
            'selbstverständlich', 'natürlich', 'offensichtlich',
            'definitiv', 'zweifellos', 'fraglos'
        } # entfernt: 'normalerweise', 'üblicherweise', 'typischerweise', 'gewöhnlich'
        
        self.conjunctions = {
            'und', 'oder', 'sowie', 'sowohl', 'als auch', 'weder noch',
            'beziehungsweise', 'bzw', 'respektive', 'wie auch',
            'nicht nur sondern auch', 'ebenso wie'
        }
        
        self.negations = {
            'nicht', 'nie', 'niemals', 'keine', 'kein', 'nichts',
            'niemand', 'nirgends', 'nirgendwo', 'keinesfalls', 
            'keineswegs'
        }

    def validate_items(self, items: List[str], keying: str = "positive") -> Dict[str, List[Union[str, Dict]]]:
        """
        Public method that calls the cached validation implementation
        """
        return self._validate_items_cached(items, keying)

    @st.cache_data(show_spinner=False)
    def _validate_items_cached(
        _self,  # Note the underscore prefix to prevent hashing
        items: List[str],
        keying: str
    ) -> Dict[str, List[Union[str, Dict]]]:
        """
        Cached implementation of item validation
        """
        valid_items = []
        invalid_items = []
        
        for item in items:
            validation_results = _self._validate_single_item(item, keying)
            if validation_results["valid"]:
                valid_items.append(item)
            else:
                invalid_items.append({
                    "item": item,
                    "reasons": validation_results["reasons"]
                })
                
        return {
            "valid_items": valid_items,
            "invalid_items": invalid_items
        }

    def _validate_single_item(self, item: str, keying: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate a single item against all criteria.
        
        Args:
            item: The item text to validate
            keying: Whether the item is intended to be positive or negative
            
        Returns:
            Dictionary with validation result and reasons for invalidity if any
        """
        reasons = []
        
        if not self._check_length(item):
            reasons.append("Ungültige Länge (4-25 Wörter erforderlich)")
            
        if not self._check_double_barreled(item):
            reasons.append("Doppelte Aussage gefunden")
            
        if not self._check_extremity(item):
            reasons.append("Extreme Formulierung gefunden")
            
        if not self._check_social_desirability(item):
            reasons.append("Sozial erwünschte Antworttendenzen gefunden")
            
        if keying == "negative":
            if not self._check_negative_item_quality(item):
                reasons.append("Mehrfache Verneinungen gefunden")  # Updated error message
        else:
            if self._contains_negation(item):
                reasons.append("Unnötige Verneinung in positivem Item")
                
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons
        }

    def _check_length(self, item: str) -> bool:
        """
        Check if item length is appropriate (4-25 words).
        """
        words = item.split()
        return 4 <= len(words) <= 25

    def _check_double_barreled(self, item: str) -> bool:
        """
        Check for double-barreled statements using conjunction markers.
        """
        return not any(conj in item.lower() for conj in self.conjunctions)

    def _check_extremity(self, item: str) -> bool:
        """
        Check for extreme language that could lead to response bias.
        """
        return not any(word in item.lower() for word in self.extreme_words)

    def _check_social_desirability(self, item: str) -> bool:
        """
        Check for obvious social desirability bias markers.
        """
        return not any(marker in item.lower() for marker in self.desirability_markers)

    def _check_negative_item_quality(self, item: str) -> bool:
        """
        Check if negative item is properly formulated.
        Only checks for multiple negations, allowing more flexible negative formulations.
        
        Args:
            item: The item text to check
            
        Returns:
            True if item has at most one negation, False if it has multiple negations
        """
        # Check for multiple negations
        negation_count = sum(item.lower().count(neg) for neg in self.negations)
        return negation_count <= 1

    def _contains_negation(self, item: str) -> bool:
        """
        Check if item contains any negations.
        """
        return any(neg in item.lower() for neg in self.negations)