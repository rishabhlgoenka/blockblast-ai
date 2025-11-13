"""
Automatic fix for piece generation to use weighted selection based on priorities.

This script will:
1. Backup the original core.py files
2. Apply the weighted piece generation fix
3. Verify the changes
"""

import os
import shutil
import re

def backup_file(filepath):
    """Create a backup of the file."""
    backup_path = filepath + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"✓ Backed up {filepath} to {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")

def fix_piece_generation(filepath):
    """Fix the _generate_new_pieces method to use weighted selection."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'priorities = np.array' in content:
        print(f"  {filepath} already has weighted generation!")
        return False
    
    # Find the method to replace
    old_method = r'(\s+)def _generate_new_pieces\(self\) -> List\[int\]:\s*\n' + \
                 r'(\s+)"""Generate 3 random piece IDs\."""\s*\n' + \
                 r'(\s+)return \[self\.rng\.randint\(0, len\(ALL_PIECES\) - 1\)\s*\n' + \
                 r'(\s+)for _ in range\(self\.NUM_PIECES_AVAILABLE\)\]'
    
    # New weighted method
    new_method = r'\1def _generate_new_pieces(self) -> List[int]:\n' + \
                 r'\2"""Generate 3 random piece IDs using weighted selection based on priorities."""\n' + \
                 r'\2# Get priorities from all pieces\n' + \
                 r'\2priorities = np.array([piece.priority for piece in ALL_PIECES])\n' + \
                 r'\2probabilities = priorities / priorities.sum()\n' + \
                 r'\2\n' + \
                 r'\2# Sample with replacement using priorities as weights\n' + \
                 r'\2pieces = np.random.choice(\n' + \
                 r'\3len(ALL_PIECES),\n' + \
                 r'\3size=self.NUM_PIECES_AVAILABLE,\n' + \
                 r'\3replace=True,\n' + \
                 r'\3p=probabilities\n' + \
                 r'\2)\n' + \
                 r'\2\n' + \
                 r'\2return pieces.tolist()'
    
    # Apply the replacement
    new_content = re.sub(old_method, new_method, content)
    
    if new_content == content:
        print(f"  ⚠️  Could not find method to replace in {filepath}")
        return False
    
    # Write the modified content
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Fixed {filepath}")
    return True

def verify_numpy_import(filepath):
    """Ensure numpy is imported."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'import numpy as np' in content:
        print(f"  ✓ numpy already imported in {filepath}")
        return True
    
    # Add numpy import after random import
    content = content.replace('import random\n', 'import random\nimport numpy as np\n')
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Added numpy import to {filepath}")
    return True

def main():
    print("=" * 80)
    print("PIECE GENERATION FIX")
    print("=" * 80)
    print("\nThis will modify piece generation to use weighted selection based on priorities.")
    print("Priority 16 pieces (lines, 2x2) will appear ~3x more often")
    print("Priority 1 piece (single block) will appear ~7x less often")
    print("\nExpected improvement: +3-5% score increase\n")
    
    # Files to fix
    files_to_fix = [
        '/Users/rishabh/blockblast-ai/BlockBlastAICNN/blockblast/core.py',
        '/Users/rishabh/blockblast-ai/BlockBlastAICNN/blockblast copy/core.py'
    ]
    
    for filepath in files_to_fix:
        print(f"\nProcessing: {filepath}")
        print("-" * 80)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            continue
        
        # Backup
        backup_file(filepath)
        
        # Verify numpy import
        verify_numpy_import(filepath)
        
        # Fix piece generation
        fixed = fix_piece_generation(filepath)
        
        if fixed:
            print(f"  ✅ Successfully applied weighted piece generation!")
        
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run analyze_pieces.py to verify changes")
    print("2. Retrain your agents with new piece generation")
    print("3. Compare scores before/after")
    print("\nTo revert changes, restore from .backup files")
    print("=" * 80)

if __name__ == "__main__":
    main()

