import re
import subprocess
import itertools
import os
import networkx as nx
from pdf2image import convert_from_path
from datasets import load_dataset
import base64
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor
import asyncio
import aiofiles
import random
import re


def extract_tikz_options(tikz_code, start_idx):
    """
    Extracts the options from a string starting from the first '[' after \begin{tikzpicture},
    handling nested square brackets correctly.

    Parameters:
    ----------
    tikz_code : str
        The input LaTeX code containing the TikZ diagram.
    start_idx : int
        The index to start searching for options (after \begin{tikzpicture}).

    Returns:
    -------
    tuple
        A tuple containing:
        - options (str): The extracted options string (if any).
        - end_idx (int): The index where the options parsing ended.
    """
    if tikz_code[start_idx:].lstrip()[0] != '[':
        return "", start_idx

    options = []
    depth = 0
    idx = start_idx

    while idx < len(tikz_code):
        char = tikz_code[idx]

        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1

        options.append(char)
        idx += 1

        # Stop when all opened brackets are closed
        if depth == 0:
            break

    return ''.join(options), idx

def parse_tikz_code(tikz_code):
    """
    Parses the content of a TikZ diagram within a LaTeX document.

    The function identifies the TikZ picture environment, extracts any options specified,
    and decomposes the content of the TikZ picture into separate elements based on environment
    boundaries and control flow structures like `\foreach` loops.

    Parameters:
    ----------
    tikz_code : str
        The LaTeX code containing the TikZ diagram.

    Returns:
    -------
    tuple
        A tuple containing:
        - elements (list of str): A list of extracted elements from the TikZ picture environment.
        - options (str): The options specified within the `\begin{tikzpicture}` command (if any).
    """



    options, elements, begin_commands, sum = "", [], [], 0
    #############   FIND ANY BEGIN COMMAND BETWEEN BEGIN DOCUMENT AND BEGIN TIKZPICTURE #################
    start_match = re.search(r'\\begin{document}.*?(\\begin{tikzpicture})', tikz_code, re.DOTALL)
    if not start_match:
        print("Error: No TikZ picture found.")
        return elements, options, begin_commands, sum

    # Extract options manually, handling nested square brackets
    start_idx = start_match.end()
    options, end_idx = extract_tikz_options(tikz_code, start_idx)

    print("Extracted options:", options)

    match = re.search(r'\\begin{document}(.*?)(\\begin{tikzpicture})', tikz_code, re.DOTALL)

    if match:
        # Extract the content between \begin{document} and the first \begin{tikzpicture}
        content_between = match.group(1)
        for line in content_between.splitlines():
            sum += line.count('{')
            sum -= line.count('}')
        # Step 2: Find all \begin{...} commands in this content
        all_begins = re.findall(r'\\begin\{[^}]+\}', content_between)

        print("All \\begin{...} commands between \\begin{document} and \\begin{tikzpicture}:")
        for begin_command in all_begins:
            content_inside = re.search(r'\{([^}]+)\}', begin_command)
            if content_inside:
                begin_commands.append(content_inside.group(1))
    else:
        print("No match found between \\begin{document} and \\begin{tikzpicture}.")
    


    ##############################################################################################


    ######################## GETTING THE LAST END{TIKZPICTURE} ############################
    reversed_code = tikz_code[::-1]

    # Search for the reversed \end{tikzpicture}
    end_match = re.search(r'}erutcipzkit{dne', reversed_code)
    if not end_match:
        raise ValueError("Error: No \\end{tikzpicture} found.")

    # Find the position of \end{tikzpicture} in the original string
    end_content = len(tikz_code) - end_match.end()

    if not end_match:
        print("Error: No closing \\end{tikzpicture} found.")
        return elements, options, begin_commands, sum

    tikz_content = tikz_code[end_idx:end_content-1].splitlines()

    ##############################################################################################

    print("Parsing TikZ code...")

    current_element = []    # Temporary list to build the current element.
    stack = []              # Stack to track open environments (e.g., \begin{scope}).
    foreach_depth = 0       # Counter to track nested \foreach loops.
    brace_sum = 0  # Initialize the sum to zero



    # Iterate through each line in the TikZ content.
    for i, line in enumerate(tikz_content):
        stripped_line = line.strip()

        # Count the number of opening braces `{` and closing braces `}` in the line
        open_braces = line.count('{')
        close_braces = line.count('}')
    
        # Update the sum: increase by the number of `{` and decrease by the number of `}`
        brace_sum += open_braces - close_braces

        # Detect the start of an environment with \begin{...} and push it onto the stack.
        if stripped_line.startswith(r'\begin{') or stripped_line.startswith(r'\scope'):
            stack.append(stripped_line)  # Store the environment name on the stack.

        # Detect the start of a \foreach loop and increment the depth counter.
        if stripped_line.startswith(r'\foreach'):
            foreach_depth += 1

        # Add the current line to the current element.
        current_element.append(line)

        # Detect the end of an environment with \end{...} and pop it from the stack.
        if stripped_line.startswith(r'\end{') or stripped_line.startswith(r'\endscope'):
            if stack:
                stack.pop()

            if brace_sum == 0 and not stack:
                elements.append('\n'.join(current_element))
                current_element = []  # Reset the current element list.

        # Decrease the \foreach depth counter when encountering a closing brace at the end of a line.
        if foreach_depth > 0 and stripped_line.endswith('}'):
            foreach_depth -= 1

        # An element is considered complete if the stack and \foreach depth are both empty,
        # and the line ends with a semicolon (common in TikZ commands).
        if not stack and foreach_depth == 0 and stripped_line.endswith(';'):
            # Join the lines of the current element and add it to the elements list.
            if brace_sum == 0:
                elements.append('\n'.join(current_element))
                current_element = []  # Reset the current element list.

    # Add any remaining lines as an element if the current element list is not empty.
    if current_element:
        elements.append('\n'.join(current_element))

    # Output the number of elements found and return them along with the options.
    print(f"Found {len(elements)} elements in the TikZ code.")
    return elements, options, begin_commands, sum



def identify_dependencies(elements):
    print("Identifying dependencies between elements...")
    G = nx.DiGraph()
    
    for i, element in enumerate(elements):
        G.add_node(i)
        for j, other_element in enumerate(elements):
            if i != j and depends_on(element, other_element):
                G.add_edge(j, i)
    
    print(f"Created dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def depends_on(element, other_element):
    # Extract node names and coordinates from both elements
    element_names = re.findall(r'\(([\w\d]+)\)', element)
    other_names = re.findall(r'\(([\w\d]+)\)', other_element)
    element_coords = re.findall(r'\(([-\d.]+,[-\d.]+)\)', element)
    other_coords = re.findall(r'\(([-\d.]+,[-\d.]+)\)', other_element)
    
    # Check if this element references any names or coordinates from the other element
    return any(name in element for name in other_names) or any(coord in element for coord in other_coords)

async def generate_valid_combinations(elements, dep_graph, code, png_folder, code_folder, output_folder):
    print("Generating valid combinations of elements...")
    valid_combinations = []
    for r in range(1, len(elements) + 1):
        for combination in itertools.combinations(range(len(elements)), r):
            if is_valid_combination(combination, dep_graph):
                valid_combinations.append(combination)
                i = len(valid_combinations)
                print(f"\nProcessing combination {i+1} of {len(valid_combinations)}...")
        
                if 'end{tikzpicture}' not in elements[combination[-1]]:
                    continue
                reconstructed_code = reconstruct_tikz_code(combination, elements, code)
                output_filename = os.path.join(png_folder, f'combination_{i+1}.png')
                
                success = await asyncio.to_thread(tikz_to_png, reconstructed_code, output_filename, output_folder)
                
                # Save the code, regardless of whether it worked or not
                code_filename = os.path.join(code_folder, f'combination_{i+1}.tex')
                async with aiofiles.open(code_filename, 'w') as f:
                    await f.write(reconstructed_code)
                
                if success:
                    print(f"Generated combination_{i+1}.png")
                    print(f"Saved working code to {code_filename}")
                else:
                    print(f"Failed to generate combination_{i+1}.png")
                    print(f"Saved non-working code to {code_filename}")
                    # Remove failed PNG if it exists
                    png_filename = os.path.join(png_folder, f'combination_{i+1}.png')
                    if os.path.exists(png_filename):
                        os.remove(png_filename)
    
    print(f"Generated {len(valid_combinations)} valid combinations.")
    return valid_combinations


async def generate_sequential_combinations(elements, code, png_folder, code_folder, output_folder, options, begin_commands, sum, max_combinations=10):
    print("Generating sequential combinations of elements with a maximum threshold...")
    valid_combinations = []
    total_elements = len(elements)
    
    # Determine the subset sizes to process
    if total_elements > max_combinations:
        # Randomly sample `max_combinations` distinct subset sizes from 1 to total_elements
        subset_sizes = sorted(random.sample(range(1, total_elements + 1), max_combinations))
    else:
        # Use all sizes if the total number is within the limit
        subset_sizes = list(range(1, total_elements + 1))

    print(f"Subset sizes selected: {subset_sizes}")

    # Iterate over the sampled subset sizes
    for r in subset_sizes:
        combination = list(range(r))  # Sequential combination
        valid_combinations.append(combination)
        i = len(valid_combinations)
        print(f"\nProcessing combination {i} of {len(subset_sizes)}...")

        # Check if the last element contains 'end{tikzpicture}' or 'end{figure}'
        if 'end{tikzpicture}' in elements[combination[-1]] or 'end{figure}' in elements[combination[-1]]:
            print(f'Combination {i} is repeated or contains an end command, skipping.')
            continue

        # Reconstruct the TikZ code for the current combination
        reconstructed_code = reconstruct_tikz_code(combination, elements, code, options, begin_commands, sum)
        output_filename = os.path.join(png_folder, f'combination_{i}.png')

        # Generate the PNG image asynchronously
        success = await asyncio.to_thread(tikz_to_png, reconstructed_code, output_filename, output_folder)

        # Save the reconstructed code, regardless of success
        code_filename = os.path.join(code_folder, f'combination_{i}.tex')
        async with aiofiles.open(code_filename, 'w') as f:
            await f.write(reconstructed_code)

        if not success:
            print(f"Failed to generate combination_{i}.png")
            print(f"Saved non-working code to {code_filename}")
            # Remove the failed PNG file if it exists
            if os.path.exists(output_filename):
                os.remove(output_filename)

        # Stop if we have reached the maximum threshold of combinations
        if len(valid_combinations) >= max_combinations:
            print(f"Reached the maximum of {max_combinations} combinations.")
            return valid_combinations

    print(f"Generated {len(valid_combinations)} sequential combinations.")
    return valid_combinations


def is_valid_combination(combo, dep_graph):
    subgraph = dep_graph.subgraph(combo)
    return nx.is_weakly_connected(subgraph) and len(list(nx.simple_cycles(subgraph))) == 0


def reconstruct_tikz_code(combination, elements, original_code, tikz_options="", begin_commands = [], sum = 0):
    # Extract everything from the start of the document up to the first \begin{tikzpicture} after \begin{document}
    match = re.search(r'^(.*?\\begin{document}.*?)(\\begin{tikzpicture})', original_code, re.DOTALL)
    
    if match:
        # Preamble includes everything up to the first \begin{tikzpicture}
        preamble = match.group(1)
    else:
        raise ValueError("The original code does not contain a valid \\begin{document} or \\begin{tikzpicture}.")

    # Add tikz_options to the \begin{tikzpicture} command if any
    options_str = f"{tikz_options}" if tikz_options else ""
    options_str = options_str.strip()
    reconstructed_tikz = f'\\begin{{tikzpicture}}{options_str}\n'
    reconstructed_tikz += '\n'.join(elements[i] for i in combination)
    reconstructed_tikz += '\n\\end{tikzpicture}'

    # Combine the preamble and the reconstructed TikZ code
    begin_commands = begin_commands[::-1]
    for begin_command in begin_commands:
        reconstructed_tikz += '\n\\end{' + begin_command + '}'
    reconstructed_tikz += '\n' + '}'*sum   
    reconstructed = f"{preamble}\n\n{reconstructed_tikz}\n\\end{{document}}"
    return reconstructed


def tikz_to_png(tikz_code, output_filename, output_folder):
    # print(f"Converting TikZ code to PNG: {output_filename}")
    temp_tex = os.path.join(output_folder, 'temp.tex')
    with open(temp_tex, 'w') as f:
        f.write(tikz_code)
    
    try:
        # Compile LaTeX to PDF
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', output_folder, temp_tex], 
                                capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            # print("LaTeX compilation failed. Error message:")
            # print(result.stdout)
            # print(result.stderr)
            return False
        
        # Convert PDF to PNG using pdf2image
        temp_pdf = os.path.join(output_folder, 'temp.pdf')
        images = convert_from_path(temp_pdf, dpi=300)
        if images:
            images[0].save(output_filename, 'PNG')
            # print(f"Conversion complete: {output_filename}")
            return True
        else:
            print("Failed to convert PDF to PNG: No images generated")
            return False
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False
    finally:
        for ext in ['.aux', '.log', '.pdf', '.tex']:
            try:
                os.remove(os.path.join(output_folder, f'temp{ext}'))
            except FileNotFoundError:
                pass


async def save_image(image, path):
# def save_image(image, path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, image.save, path)
    # image.save(path)

async def process_tikz_diagram(code, image, output_folder, filename, sequential):
# def process_tikz_diagram(code, image, output_folder, filename, sequential):
    print(f"\nProcessing TikZ diagram from {filename}...")
    
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    png_folder = os.path.join(output_folder, 'png')
    os.makedirs(png_folder, exist_ok=True)
    code_folder = os.path.join(output_folder, 'code')
    os.makedirs(code_folder, exist_ok=True)
    
    # Save the original image
    main_output_filename = os.path.join(png_folder, 'main.png')
    await save_image(image, main_output_filename)
    # print(f"Saved original image: {main_output_filename}")
    
    # Save the original code
    original_code_filename = os.path.join(code_folder, 'original.tex')
    async with aiofiles.open(original_code_filename, 'w') as f:
        await f.write(code)
    # print(f"Saved original code to {original_code_filename}")
    
    elements, options, begin_commands, sum = parse_tikz_code(code)
    dep_graph = identify_dependencies(elements)
    if sequential:
        valid_combinations = await generate_sequential_combinations(elements, code, png_folder, code_folder, output_folder, options, begin_commands, sum)
    else:
        valid_combinations = await generate_valid_combinations(elements, dep_graph, code, png_folder, code_folder, output_folder)
    
    # for i, combination in enumerate(valid_combinations):
        

    print(f"\nProcessing complete for {filename}!")

async def process_huggingface_dataset(dataset_name, output_base_directory, sequential, start_index, end_index, num_examples=100):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    # examples = [13372]
    for i in range(start_index, end_index):
    # for i in examples:
        sample = dataset['train'][i]
        code = sample["code"]
        image = sample["image"]
        filename = f"example_{i}"
        
        output_folder = os.path.join(output_base_directory, filename)
        if os.path.exists(output_folder):
            print(f"skippong example_{i} already exists")
            continue
        await process_tikz_diagram(code, image, output_folder, filename, sequential)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process TikZ images and code.")
    parser.add_argument('--start_index', type=int, default=0, help="Start index for processing.")
    parser.add_argument('--end_index', type=int, default=None, help="End index for processing.")
    args = parser.parse_args()

    # Example usage
    dataset_name = "nllg/datikz-v2"
    output_base_directory = "tikz_decomposition_output"
    sequential = True
    if sequential:
        output_base_directory+="_sequential"
    asyncio.run(process_huggingface_dataset(dataset_name, output_base_directory, sequential, start_index=args.start_index, end_index=args.end_index))

