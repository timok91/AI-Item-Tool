# item_generation/item_generator.py

from typing import Dict, List, Union
import streamlit as st
from anthropic import Anthropic
from .item_validator import GermanItemValidator
import re

class GermanPersonalityItemGenerator:
    """
    Generates German personality test items using Claude API.
    Implements evidence-based best practices for personality item generation
    and integrates with semantic similarity analysis.
    """
    
    def __init__(self, anthropic_client: Anthropic):
        self.client = anthropic_client
        self.validator = GermanItemValidator()

    def generate_items(self,
                  construct_definition: str,
                  n_items: int = 10,
                  work_context: bool = False,
                  negative_ratio: float = 0.3,
                  model: str = "claude-3-haiku-20240307",  
                  temperature: float = 0.7) -> Dict[str, Union[List[str], Dict]]:
        """
        Generate personality items with controlled positive/negative ratio.
        """
        # Move the caching to a separate internal method
        return self._generate_items_cached(
            construct_definition,
            n_items,
            work_context,
            negative_ratio,
            model,
            temperature
        )

    @st.cache_data(show_spinner=False)
    def _generate_items_cached(
        _self,  # Note the underscore prefix to prevent hashing
        construct_definition: str,
        n_items: int,
        work_context: bool,
        negative_ratio: float,
        model: str,
        temperature: float
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Cached version of generate_items implementation
        """
        if not 0.0 <= negative_ratio <= 0.5:
            raise ValueError("negative_ratio must be between 0.0 and 0.5")
            
        n_negative = int(n_items * negative_ratio)
        n_positive = n_items - n_negative
        
        # Generate 20% more items to account for invalid ones
        n_positive_generate = int(n_positive * 1.2)
        n_negative_generate = int(n_negative * 1.2)
        
        # Generate and validate items
        positive_items = _self._generate_items(
            construct_definition, 
            n_positive_generate, 
            work_context,
            "positive",
            model,
            temperature
        )
        
        negative_items = _self._generate_items(
            construct_definition, 
            n_negative_generate, 
            work_context,
            "negative",
            model,
            temperature
        )
        
        # Trim to requested numbers if we have more valid items than needed
        positive_valid = positive_items["valid_items"][:n_positive]
        negative_valid = negative_items["valid_items"][:n_negative]
        
        # Combine all items for embedding analysis
        all_valid_items = positive_valid + negative_valid
        
        return {
            "positive": positive_valid,
            "negative": negative_valid,
            "invalid": {
                "positive": positive_items["invalid_items"],
                "negative": negative_items["invalid_items"]
            },
            "all_items": all_valid_items,
            "metadata": {
                "construct": construct_definition,
                "work_context": work_context,
                "n_requested": n_items,
                "n_generated": len(all_valid_items),
                "model": model,            
                "temperature": temperature
            }
        }

    def _generate_items(self, 
                   construct_def: str, 
                   n_items: int, 
                   work_context: bool,
                   keying: str,
                   model: str,
                   temperature: float) -> Dict[str, List[str]]:
        """
        Generate and validate items for a specific keying.
        """
        prompt = self._create_prompt(construct_def, n_items, work_context, keying)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract the actual content from the response
            content = response.content[0].text if hasattr(response, 'content') else str(response)
            items = self._parse_items(content)
            return self.validator.validate_items(items, keying)
            
        except Exception as e:
            st.error(f"Fehler bei der Item-Generierung: {str(e)}")
            return {"valid_items": [], "invalid_items": []}

    def _create_prompt(self, construct_def: str, n_items: int, work_context: bool, keying: str) -> str:
        """
        Create a detailed prompt for German personality item generation with specified keying.
        """
        base_prompt = f"""Als erfahrener Persönlichkeitspsychologe generieren Sie {n_items} {'positiv' if keying == 'positive' else 'negativ'} 
formulierte Testitems zur Messung des folgenden Konstrukts:

KONSTRUKT-DEFINITION:
{construct_def}

WISSENSCHAFTLICHE ANFORDERUNGEN:
1. Jedes Item muss eine Aussage in der ersten Person Singular sein, muss aber nicht zwingend mit Ich beginnen
2. Items müssen das gesamte Spektrum des Konstrukts abdecken
3. Formulierungen müssen klar, präzise und ohne Fachbegriffe sein
4. Items müssen für eine 5-stufige Likert-Skala (stimme gar nicht zu - stimme voll zu) geeignet sein
5. Vermeiden Sie:
   - Extreme Formulierungen (z.B. 'immer', 'nie', 'alle')
   - Sozial erwünschte Antworttendenzen
   - Doppelte Aussagen in einem Item
   - Komplexe Verneinungen
6. Jedes Item muss eine klar beobachtbare Verhaltenstendenz oder Einstellung erfassen
7. Items sollen optimal zwischen Personen differenzieren können
8. Items müssen für die allgemeine erwachsene Bevölkerung verständlich sein
9. Items sollen möglichst konkret formuliert sein

Reduziere sozial erwünschte Antworten insbesondere durch die Methode der evaluativen Neutralisierung:
Die evaluative Neutralisierung beinhaltet die Umformulierung von Items in eine neutralere Form, wodurch sozial erwünschte Antworten weniger wahrscheinlich werden. Der negative Klang von „Kontakt mit anderen vermeiden" kann abgemildert werden, indem man es zu „Fühle mich auch allein wohl" umformuliert. Ähnlich kann ein positives Item wie „Mag Ordnung" neutraler ausgedrückt werden als „Bin nur zufrieden, wenn die Dinge systematisch geordnet sind."
Die evaluative Neutralisierung zielt darauf ab, den bewertenden Gehalt von Items zu reduzieren und dabei die inhaltlichen Aspekte des relevanten Merkmals beizubehalten, um dadurch die Kriteriumsvalidität zu verbessern (Leising, Burger, et al., 2020). Dies kann durch umfangreiche Umformulierung der Items erreicht werden oder manchmal durch den Austausch eines einzelnen Eigenschaftsadjektivs im Item durch ein weniger wertendes. Peabody (1967, 1984) und Borkenau und Ostendorf (1989) liefern verschiedene Beispiele für Adjektive mit ähnlicher Bedeutung, aber deutlich unterschiedlicher Wertigkeit (z.B. skeptisch versus misstrauisch, wählerisch versus pingelig und bestimmt versus streng).
"""

        if keying == "positive":
            base_prompt += """

BEISPIELE FÜR POSITIVE ITEMS:
- Ich erledige meine Aufgaben sehr sorgfältig
- Ich plane meine Aktivitäten genau im Voraus
- Details nehme ich sehr ernst bei meiner Arbeit"""
        else:
            base_prompt += """

BEISPIELE FÜR NEGATIVE ITEMS:
- Ich erledige Aufgaben oft erst auf den letzten Drücker
- Genaue Planung ist mir zu aufwendig
- Details sind mir meist nicht so wichtig

Vermeiden Sie dabei doppelte Verneinungen. Nutzen Sie stattdessen Formulierungen wie:
- 'selten', 'ungern', 'wenig'
- 'fällt mir schwer', 'bereitet mir Mühe'
- 'ist mir nicht wichtig', 'liegt mir nicht'"""

        if work_context:
            base_prompt += """

BERUFLICHER KONTEXT:
Formulieren Sie die Items im Arbeitskontext. Beziehen Sie sich auf:
- Typische Arbeitssituationen
- Berufliche Aufgaben und Herausforderungen
- Interaktionen mit Kollegen und Vorgesetzten
- Berufliche Leistung und Erfolg"""

        base_prompt += """

FORMAT:
- Verwenden Sie Präsens
- Achten Sie auf korrekte deutsche Grammatik und Rechtschreibung
- Listen Sie die Items einfach durchnummeriert auf
- Nennen Sie nur die tatsächlichen Items und NIEMALS irgendwelche Einleitungssätze oder zusätzlichen Erklärungen"""

        return base_prompt

    def _parse_items(self, response_text: str) -> List[str]:
        """
        Parse the response text to extract individual items.
        
        Args:
            response_text: Raw response text from Claude
            
        Returns:
            List of individual items
        """
        items = []
        
        try:
            # Split text into lines and process each line
            lines = str(response_text).split('\n')
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Clean the line
                cleaned = re.sub(r'^\d+\.?\s*', '', line.strip())
                
                # Skip lines that don't look like items
                if len(cleaned) < 10 or not any(c.isalpha() for c in cleaned):
                    continue
                    
                # Ensure proper punctuation
                if not cleaned.endswith('.'):
                    cleaned += '.'
                    
                items.append(cleaned)
                    
        except Exception as e:
            st.error(f"Fehler beim Parsen der Items: {str(e)}")
            return []

        return items

    def format_results_for_display(self, generated_items: Dict) -> None:
        """
        Display generated items in a formatted way using Streamlit.
        """
        if not generated_items:
            st.warning("Keine Items generiert.")
            return

        # Display valid items
        st.subheader("Generierte Items")
        
        # Positive items
        if generated_items.get("positive"):
            st.write("**Positiv formulierte Items:**")
            for i, item in enumerate(generated_items["positive"], 1):
                st.write(f"{i}. {item}")
            st.write("")  # Add spacing

        # Negative items
        if generated_items.get("negative"):
            st.write("**Negativ formulierte Items:**")
            for i, item in enumerate(generated_items["negative"], 1):
                st.write(f"{i}. {item}")
            st.write("")  # Add spacing

        # Display invalid items if any exist
        invalid_positive = generated_items.get("invalid", {}).get("positive", [])
        invalid_negative = generated_items.get("invalid", {}).get("negative", [])
        
        if invalid_positive or invalid_negative:
            st.write("**❌ Verworfene Items:**")
            if invalid_positive:
                st.write("*Verworfene positive Items:*")
                for item in invalid_positive:
                    st.write(f"- {item['item']}")
                    st.write(f"  *Gründe: {', '.join(item['reasons'])}*")
                st.write("")
            
            if invalid_negative:
                st.write("*Verworfene negative Items:*")
                for item in invalid_negative:
                    st.write(f"- {item['item']}")
                    st.write(f"  *Gründe: {', '.join(item['reasons'])}*")

        # Display metadata
        if "metadata" in generated_items:
            st.write("**ℹ️ Generierungs-Details:**")
            meta = generated_items["metadata"]
            st.write(f"- Angeforderte Items: {meta.get('n_requested', 'N/A')}")
            st.write(f"- Generierte Items: {meta.get('n_generated', 'N/A')}")
            st.write(f"- Arbeitskontext: {'Ja' if meta.get('work_context') else 'Nein'}")
            st.write(f"- Modell: {meta.get('model', 'N/A')}")
            st.write(f"- Temperature: {meta.get('temperature', 'N/A')}")