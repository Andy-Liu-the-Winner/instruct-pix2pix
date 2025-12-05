"""
TORE: Transforming Original Relations Effectively

From ESPLoRA paper (arXiv:2504.13745):
T2I models exhibit systematic biases - they perform better on certain spatial
relationships (top, left, front) than their counterparts (bottom, right, behind).

TORE is a pre-processing step that exploits these biases by flipping prompts:
- "A bottom B" -> "B top A"
- "A right B" -> "B left A"
- "A behind B" -> "B in front of A"

This maintains the same semantic meaning while using relationships the model handles better.
"""

import re
from typing import Tuple, Optional


# Relationship pairs: (weak, strong)
# Model is weaker at the first, stronger at the second
RELATIONSHIP_PAIRS = [
    # 2D spatial
    ("bottom", "top"),
    ("below", "above"),
    ("under", "over"),
    ("right", "left"),
    ("to the right of", "to the left of"),
    ("on the right of", "on the left of"),
    # 3D spatial
    ("behind", "in front of"),
    ("in back of", "in front of"),
    ("at the back of", "at the front of"),
]

# Precompile regex patterns for efficiency
PATTERNS = []
for weak, strong in RELATIONSHIP_PAIRS:
    # Pattern: "A <weak> B" or "A <weak> the B"
    # Captures: group(1)=A, group(2)=B
    pattern = re.compile(
        rf"(.+?)\s+{re.escape(weak)}\s+(?:the\s+)?(.+)",
        re.IGNORECASE
    )
    PATTERNS.append((pattern, weak, strong))


def apply_tore(prompt: str) -> Tuple[str, bool]:
    """
    Apply TORE transformation to a prompt.

    Flips spatial relationships from weak (bottom/right/behind)
    to strong (top/left/front) versions.

    Args:
        prompt: Original text prompt

    Returns:
        Tuple of (transformed_prompt, was_transformed)

    Examples:
        >>> apply_tore("A cat below a table")
        ("A table above a cat", True)

        >>> apply_tore("A dog to the right of a tree")
        ("A tree to the left of a dog", True)

        >>> apply_tore("A bird behind a house")
        ("A house in front of a bird", True)

        >>> apply_tore("A cat on a mat")  # No weak relationship
        ("A cat on a mat", False)
    """
    original = prompt

    for pattern, weak, strong in PATTERNS:
        match = pattern.match(prompt)
        if match:
            obj_a = match.group(1).strip()
            obj_b = match.group(2).strip()

            # Flip: "A weak B" -> "B strong A"
            transformed = f"{obj_b} {strong} {obj_a}"

            # Preserve any trailing context (e.g., "in a city")
            # Check if there's more text after the match
            remaining = prompt[match.end():]
            if remaining.strip():
                transformed += remaining

            return transformed, True

    return prompt, False


def apply_tore_batch(prompts: list) -> Tuple[list, int]:
    """
    Apply TORE to a batch of prompts.

    Args:
        prompts: List of text prompts

    Returns:
        Tuple of (transformed_prompts, num_transformed)
    """
    transformed = []
    count = 0

    for prompt in prompts:
        new_prompt, was_transformed = apply_tore(prompt)
        transformed.append(new_prompt)
        if was_transformed:
            count += 1

    return transformed, count


class TOREPreprocessor:
    """
    TORE preprocessor for inference pipelines.

    Usage:
        tore = TOREPreprocessor()

        # Single prompt
        prompt = tore(prompt)

        # Or with logging
        prompt, changed = tore.transform(prompt)
        if changed:
            print(f"TORE: {original} -> {prompt}")
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: If True, print transformations
        """
        self.verbose = verbose
        self.transform_count = 0
        self.total_count = 0

    def __call__(self, prompt: str) -> str:
        """Apply TORE and return transformed prompt."""
        new_prompt, _ = self.transform(prompt)
        return new_prompt

    def transform(self, prompt: str) -> Tuple[str, bool]:
        """
        Apply TORE transformation.

        Args:
            prompt: Original prompt

        Returns:
            Tuple of (transformed_prompt, was_transformed)
        """
        self.total_count += 1
        new_prompt, was_transformed = apply_tore(prompt)

        if was_transformed:
            self.transform_count += 1
            if self.verbose:
                print(f"[TORE] '{prompt}' -> '{new_prompt}'")

        return new_prompt, was_transformed

    def get_stats(self) -> dict:
        """Return transformation statistics."""
        return {
            "total": self.total_count,
            "transformed": self.transform_count,
            "percentage": 100 * self.transform_count / max(1, self.total_count)
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.transform_count = 0
        self.total_count = 0


# Quick test
if __name__ == "__main__":
    test_prompts = [
        "A cat below a table",
        "A dog to the right of a tree",
        "A bird behind a house",
        "A person under an umbrella in a city",
        "A ball to the right of a box in a street",
        "A car behind a building in downtown",
        "A lamp above a desk",  # Already strong, no change
        "Put a hat on the cat",  # No spatial relationship
        "A bench to the left of a statue",  # Already strong
    ]

    print("TORE Transformation Examples:")
    print("=" * 60)

    tore = TOREPreprocessor(verbose=True)

    for prompt in test_prompts:
        result = tore(prompt)
        if result != prompt:
            print(f"  Original:    {prompt}")
            print(f"  Transformed: {result}")
            print()

    stats = tore.get_stats()
    print(f"\nStats: {stats['transformed']}/{stats['total']} prompts transformed ({stats['percentage']:.1f}%)")
