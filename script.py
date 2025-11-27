#!/usr/bin/env python3
"""
Script to recursively read all files in a directory and write their contents
to a text file with relative paths in lexicographical order.
Only includes coding language files. Converts HTML to markdown format.
"""

import os
import sys
from pathlib import Path
import re


def html_to_markdown(html_content):
    """
    Convert HTML content to markdown format by cleaning tags.
    
    Args:
        html_content: String containing HTML content
        
    Returns:
        String with markdown-formatted content
    """
    # Remove script and style tags with their content
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert headings
    html_content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<h5[^>]*>(.*?)</h5>', r'##### \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<h6[^>]*>(.*?)</h6>', r'###### \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert bold and strong
    html_content = re.sub(r'<(b|strong)[^>]*>(.*?)</\1>', r'**\2**', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert italic and emphasis
    html_content = re.sub(r'<(i|em)[^>]*>(.*?)</\1>', r'*\2*', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert links
    html_content = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert images
    html_content = re.sub(r'<img[^>]*src=["\']([^"\']*)["\'][^>]*alt=["\']([^"\']*)["\'][^>]*/?>', r'![\2](\1)', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*src=["\']([^"\']*)["\'][^>]*/?>', r'![\1](\2)', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<img[^>]*src=["\']([^"\']*)["\'][^>]*/?>', r'![](\1)', html_content, flags=re.IGNORECASE)
    
    # Convert code blocks
    html_content = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert inline code
    html_content = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert blockquotes
    html_content = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', lambda m: '\n'.join('> ' + line for line in m.group(1).strip().split('\n')), html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert unordered lists
    html_content = re.sub(r'<ul[^>]*>(.*?)</ul>', lambda m: convert_list(m.group(1), '-'), html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert ordered lists
    html_content = re.sub(r'<ol[^>]*>(.*?)</ol>', lambda m: convert_list(m.group(1), '1.'), html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert line breaks
    html_content = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)
    
    # Convert paragraphs
    html_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert divs and spans (just remove tags, keep content)
    html_content = re.sub(r'<div[^>]*>(.*?)</div>', r'\1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove all remaining HTML tags
    html_content = re.sub(r'<[^>]+>', '', html_content)
    
    # Decode HTML entities
    html_content = html_content.replace('&nbsp;', ' ')
    html_content = html_content.replace('&lt;', '<')
    html_content = html_content.replace('&gt;', '>')
    html_content = html_content.replace('&amp;', '&')
    html_content = html_content.replace('&quot;', '"')
    html_content = html_content.replace('&#39;', "'")
    
    # Clean up excessive whitespace
    html_content = re.sub(r'\n\s*\n\s*\n', '\n\n', html_content)
    html_content = html_content.strip()
    
    return html_content


def convert_list(list_content, marker):
    """
    Convert HTML list items to markdown format.
    
    Args:
        list_content: String containing list items
        marker: List marker (- for unordered, 1. for ordered)
        
    Returns:
        String with markdown-formatted list
    """
    items = re.findall(r'<li[^>]*>(.*?)</li>', list_content, flags=re.DOTALL | re.IGNORECASE)
    result = []
    for i, item in enumerate(items):
        item = item.strip()
        if marker == '1.':
            result.append(f'{i+1}. {item}')
        else:
            result.append(f'{marker} {item}')
    return '\n'.join(result) + '\n'


def get_all_files(directory):
    """
    Recursively get all files in a directory in lexicographical order.
    Only includes files with coding language extensions.
    
    Args:
        directory: Path object of the directory to traverse
        
    Returns:
        List of Path objects for all files, sorted lexicographically
    """
    # Define allowed coding file extensions
    allowed_extensions = {
        '.py'
    }
    
    all_files = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(directory):
        # Sort directories and files to ensure lexicographical order
        dirs.sort()
        files.sort()
        
        # Add all files from current directory that match allowed extensions
        for file in files:
            file_path = Path(root) / file
            # Check if file extension is in allowed list
            if file_path.suffix.lower() in allowed_extensions:
                all_files.append(file_path)
    
    return all_files


def write_files_to_output(directory_path, output_file):
    """
    Write all files from directory to output file with their relative paths.
    
    Args:
        directory_path: Path to the input directory
        output_file: Path to the output text file
    """
    directory = Path(directory_path).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        sys.exit(1)
    
    # Get the script's own path to exclude it
    script_path = Path(__file__).resolve()
    
    # Get all files in lexicographical order
    files = get_all_files(directory)
    
    # Filter out the script file itself
    files = [f for f in files if f.resolve() != script_path]
    
    if not files:
        print(f"Warning: No files found in '{directory_path}'")
        return
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, file_path in enumerate(files):
            # Get relative path from the input directory
            relative_path = file_path.relative_to(directory)
            
            # Write the relative path
            outfile.write(f"{relative_path}\n\n")
            
            # Try to read and write file contents
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                    # Convert HTML to markdown if it's an HTML file
                    if file_path.suffix.lower() == '.html':
                        content = html_to_markdown(content)
                    
                    outfile.write(content)
            except UnicodeDecodeError:
                # Handle binary files
                outfile.write("[Binary file - content not displayed]\n")
            except Exception as e:
                outfile.write(f"[Error reading file: {e}]\n")
            
            # Add separator between files (except after the last file)
            if i < len(files) - 1:
                outfile.write("\n\n")
    
    print(f"Successfully wrote {len(files)} files to '{output_file}'")


def main():
    
    directory_path = "/Users/arnav/Desktop/dist/"
    
    # Use default output filename if not provided
    output_file = "output.txt"
    
    write_files_to_output(directory_path, output_file)


if __name__ == "__main__":
    main()