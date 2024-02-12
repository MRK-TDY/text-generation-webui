import yaml
import colorama
import os


# https://docs.gpt4all.io/gpt4all_python.html
# https://github.com/nomic-ai/gpt4all

class PalAI():
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'prompts.yaml'), 'r') as file:
            self.prompts_file = yaml.safe_load(file)
        self.system_prompt = self.prompts_file['system_prompt']
        self.prompt_template = self.prompts_file['prompt_template']

    def format_prompt(self, user_prompt):
        return (self.system_prompt, self.prompt_template.format(user_prompt))

    def extract_building_information(self, text):
        lines = text.split('\n')
        building_info = []

        # match lines that have two `|` characters
        for line in lines:
            if line.count('|') == 2:
                building_info.append(line)

        blocks = []
        for block in building_info:
            block = block.split('|')
            blocks.append({'type': block[0], 'position': f"({block[1]})", 'size': block[2]})

        return blocks

    def generate_obj(self, api_response):
        """
        Takes an API response and generates an OBJ file representation.

        Parameters:
        - api_response: A list of dictionaries, where each dictionary represents a block.

        Returns:
        - A string representing the content of an OBJ file.
        """
        block_obj_paths = {
            'CUBE': 'Blocks/Cube_Block.obj',
            'CHIPPED_CUBE': 'Blocks/Cube_Block.obj',
            'CONCAVE_CUBE': 'Blocks/Cube_Block.obj'
            # Add more block types here
        }

        obj_content = ''
        vertex_offset = 0  # Initialize vertex offset

        for block in api_response:
            if block['type'] not in block_obj_paths:
                print("Block type not implemented:", block['type'])
                block['type'] = 'CUBE'  # Default to a cube if the block type is not implemented

            try:
                block_name = block['type']
                position = tuple(map(float, block['position'].replace("(", "").replace(")", "").split(',')))
                size = float(block['size'])
            except:
                print("Invalid block data:", block)
                continue

            path = os.path.join(os.path.dirname(__file__), block_obj_paths[block_name])
            # Load the template OBJ for this block
            with open(path) as file:
                block_obj = file.read()

            # Adjust the vertices based on position and size
            for line in block_obj.splitlines():
                if line.startswith('v '):  # Vertex definition
                    parts = line.split()
                    x, y, z = [float(parts[1]) * size + position[0],
                               float(parts[2]) * size + position[1],
                               float(parts[3]) * size + position[2]]
                    obj_content += f'v {x} {y} {z}\n'
                elif line.startswith('f '):  # Face definition, adjust with the current vertex offset
                    parts = line.split()[1:]
                    faces = []
                    for f in parts:
                        face = f.split('/')
                        face[0] = str(int(face[0]) + vertex_offset)
                        faces.append('/'.join(face))
                    obj_content += 'f ' + ' '.join(faces) + '\n'
                elif line.startswith('vt '):
                    obj_content += line + '\n'
                elif line.startswith('vn '):
                    obj_content += line + '\n'

            # Update the vertex offset for the next block
            vertex_offset += sum(1 for line in block_obj.splitlines() if line.startswith('v '))

        return obj_content