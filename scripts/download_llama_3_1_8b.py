#!/usr/bin/env python3
"""
Download Llama 3.1 8B Instruct Model from HuggingFace

This script downloads the Llama-3.1-8B-Instruct model with proper authentication.
The model requires accepting the license on HuggingFace and using an access token.

Usage:
    python scripts/download_llama_3_1_8b.py [--output_dir models/llama-3.1-8b-instruct]

Requirements:
    - HuggingFace account with Llama 3.1 access approved
    - HuggingFace token with read permissions
    
Setup:
    1. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    2. Create token: https://huggingface.co/settings/tokens
    3. Login: huggingface-cli login
    Or set environment variable: export HF_TOKEN="your_token_here"
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login, HfFolder
except ImportError:
    print("❌ Missing required packages. Please install:")
    print("   pip install transformers huggingface-hub")
    sys.exit(1)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def check_authentication():
    """Check if user is authenticated with HuggingFace."""
    token = HfFolder.get_token()
    
    if token is None:
        # Check environment variable
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        if token is None:
            console.print("\n[bold red]❌ HuggingFace authentication required![/bold red]\n")
            console.print("Please authenticate using ONE of these methods:\n")
            console.print("1. [bold cyan]Interactive login:[/bold cyan]")
            console.print("   huggingface-cli login\n")
            console.print("2. [bold cyan]Environment variable:[/bold cyan]")
            console.print("   export HF_TOKEN='your_token_here'\n")
            console.print("3. [bold cyan]Python login:[/bold cyan]")
            console.print("   from huggingface_hub import login")
            console.print("   login(token='your_token_here')\n")
            console.print("Get your token at: https://huggingface.co/settings/tokens")
            console.print("Request model access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct\n")
            return False
        else:
            # Login with environment token
            login(token=token)
            console.print("[green]✓ Authenticated via environment variable[/green]")
    else:
        console.print("[green]✓ Already authenticated with HuggingFace[/green]")
    
    return True


def download_model(model_name: str, output_dir: Path, force_download: bool = False):
    """
    Download Llama model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save model
        force_download: Force re-download even if model exists
    """
    
    console.print(f"\n[bold cyan]📥 Downloading {model_name}[/bold cyan]\n")
    console.print(f"Output directory: {output_dir}")
    
    # Check if model already exists
    if output_dir.exists() and not force_download:
        if (output_dir / "config.json").exists():
            console.print(f"\n[yellow]⚠ Model already exists at {output_dir}[/yellow]")
            response = input("Do you want to re-download? (y/N): ").strip().lower()
            if response != 'y':
                console.print("[green]✓ Using existing model[/green]")
                return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download tokenizer
        console.print("\n[bold]Step 1/2: Downloading tokenizer...[/bold]")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=output_dir,
            token=True  # Use authenticated token
        )
        tokenizer.save_pretrained(output_dir)
        console.print("[green]✓ Tokenizer downloaded[/green]")
        
        # Download model
        console.print("\n[bold]Step 2/2: Downloading model (~16 GB)...[/bold]")
        console.print("[dim]This may take 5-15 minutes depending on your connection...[/dim]\n")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=output_dir,
            torch_dtype="auto",  # Use appropriate dtype
            token=True,  # Use authenticated token
            low_cpu_mem_usage=True  # Optimize memory usage during download
        )
        model.save_pretrained(output_dir)
        console.print("\n[green]✓ Model downloaded successfully![/green]")
        
        # Verify download
        console.print("\n[bold]Verifying download...[/bold]")
        files_to_check = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
        
        all_present = True
        for file in files_to_check:
            if (output_dir / file).exists():
                console.print(f"  [green]✓[/green] {file}")
            else:
                console.print(f"  [red]✗[/red] {file} - MISSING")
                all_present = False
        
        if all_present:
            console.print("\n[bold green]🎉 Download complete and verified![/bold green]")
            
            # Display model info
            total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            total_size_gb = total_size / (1024 ** 3)
            
            console.print(f"\n[bold]Model Information:[/bold]")
            console.print(f"  Location: {output_dir}")
            console.print(f"  Size: {total_size_gb:.2f} GB")
            console.print(f"  Model: {model_name}")
            console.print(f"  Files: {len(list(output_dir.rglob('*')))}")
            
            # Quick usage example
            console.print("\n[bold]Quick Usage:[/bold]")
            console.print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{output_dir}",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
""")
        else:
            console.print("\n[bold red]⚠ Download incomplete - some files are missing![/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during download:[/bold red]")
        console.print(f"  {str(e)}\n")
        
        if "401" in str(e) or "403" in str(e) or "unauthorized" in str(e).lower():
            console.print("[yellow]This error suggests an authentication issue:[/yellow]")
            console.print("  1. Ensure you've requested access to the model")
            console.print("  2. Verify your HuggingFace token is valid")
            console.print("  3. Check that your token has 'read' permissions")
            console.print("\nRequest access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Llama 3.1 8B Instruct model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python scripts/download_llama_3_1_8b.py
  
  # Download to custom location
  python scripts/download_llama_3_1_8b.py --output_dir /path/to/models
  
  # Force re-download
  python scripts/download_llama_3_1_8b.py --force
  
Setup Authentication:
  1. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  2. Create token: https://huggingface.co/settings/tokens
  3. Login: huggingface-cli login
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/llama-3.1-8b-instruct",
        help="Directory to save the model (default: models/llama-3.1-8b-instruct)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    # Print header
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Llama 3.1 8B Instruct Model Downloader          ║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════╝[/bold cyan]\n")
    
    # Check authentication
    if not check_authentication():
        sys.exit(1)
    
    # Convert output_dir to Path
    output_dir = Path(args.output_dir)
    
    # Download model
    download_model(
        model_name=args.model_name,
        output_dir=output_dir,
        force_download=args.force
    )
    
    console.print("\n[bold green]✅ All done![/bold green]\n")


if __name__ == "__main__":
    main()
