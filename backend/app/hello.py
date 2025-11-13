# backend/app/hello.py

def greet(name: str = "Pratishtha"):
    """
    Simple function to test backend environment setup.
    """
    return f"Hello, {name}! 🎉 Your backend environment is ready."

if __name__ == "__main__":
    print(greet())
