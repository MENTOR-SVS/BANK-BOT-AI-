# Test the dialogue manager with various inputs
from dialogue_manager import handle_input

test_cases = [
    "card",
    "debit card",
    "block card",
    "i want to block my debit card"
]

print("\nTesting Dialogue Manager\n" + "="*50)

for test_input in test_cases:
    print(f"\nTesting input: '{test_input}'")
    print("-"*30)
    
    intent, entities, response = handle_input(test_input)
    
    print(f"Intent: {intent}")
    if entities:
        print(f"Entities: {entities}")
    print(f"Response: {response}")
    print("="*50)