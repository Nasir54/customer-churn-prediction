# template_check.py
import os

def check_templates():
    template_dir = 'app/templates'
    required_files = ['index.html', 'result.html', 'error.html']
    
    print("Checking template files...")
    print("Template directory exists:", os.path.exists(template_dir))
    
    if os.path.exists(template_dir):
        print("Files in template directory:", os.listdir(template_dir))
        
        for file in required_files:
            file_path = os.path.join(template_dir, file)
            print(f"{file} exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(f"{file} length: {len(content)} characters")
                    if '{%' in content and '%}' in content:
                        print(f"{file} contains template syntax")
                    else:
                        print(f"{file} may not have proper template syntax")

if __name__ == '__main__':
    check_templates()