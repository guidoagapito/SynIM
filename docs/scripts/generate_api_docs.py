import os
from pathlib import Path

def generate_module_doc(module_name, title=None):
    """Generate simple RST content for a module"""
    if title is None:
        title = module_name.replace('_', ' ').title()

    content = f"""{title}
{'=' * len(title)}

.. automodule:: synim.{module_name}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    return content

def generate_api_index():
    """Generate API index page"""
    content = """API Reference
=============

.. toctree::
   :maxdepth: 2

   synim
   synpm
   params_manager
   params_utils
   utils

"""
    return content

def main():
    # Base paths
    api_docs_path = Path(__file__).parent.parent / "api"
    api_docs_path.mkdir(exist_ok=True)

    # Modules to document
    modules = {
        "synim": "Interaction Matrix Module",
        "synpm": "Projection Matrix Module",
        "params_manager": "Parameters Manager",
        "params_utils": "Parameters Utilities",
        "utils": "Utility Functions"
    }

    print("Generating API documentation for SynIM...\n")

    # Generate index
    print("Generating api/index.rst...")
    with open(api_docs_path / "index.rst", 'w') as f:
        f.write(generate_api_index())
    print("  ✓ Generated\n")

    # Generate individual module docs
    for module_name, title in modules.items():
        print(f"Generating api/{module_name}.rst...")
        content = generate_module_doc(module_name, title)

        with open(api_docs_path / f"{module_name}.rst", 'w') as f:
            f.write(content)
        print("  ✓ Generated")

    print("\n" + "="*50)
    print("Done! Generated files:")
    for rst_file in sorted(api_docs_path.glob("*.rst")):
        print(f"  • {rst_file.name}")

if __name__ == "__main__":
    main()
