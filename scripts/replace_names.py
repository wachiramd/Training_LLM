
import os

# List of names to replace with "user"
names_to_replace = [
    "Sashay Joy Katuu", "Kelvin Kamau", "Kevin Mutuma", "Sienka Dounia", "M.T Tessy", 
    "Kipruto", "Em Jay", "Geoffrey Koech", "Hez Omwamba", "Emelda Kwamboka 25", 
    "Eve Wanjiku", "Elsy Jemutai", "Morgan kirui", "Mary Lemerelle", "Elsie", 
    "Ben In BITS", "Newton", "Andrew Emacar", "Peter Barasa", "Namesake", 
    "Samcollins Kariuki", "Twinny", "Robi Doreen Mwita", "Abby 25", "Stacey Chumba", 
    "Vincent PSC", "Ian Kasawa", "Lenkai Solomon", "Griffin Okemwa", "Edna Wambua", 
    "Habona Gijo", "Joan PSC", "Georgia Jeptoo", "Mellan Nekesa", "Wendy Waithira", 
    "Lulu Ilina", "Little Siz", "Sam Apples Photography", "Alvin"
]

file_path = "/Users/daniel/Documents/ILINA project/Dan_LLM/Train_Your_Language_Model_Course/output/combined_text.txt"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace names
    for name in names_to_replace:
        content = content.replace(name, "user")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully replaced names with 'user'.")

except Exception as e:
    print(f"An error occurred: {e}")
