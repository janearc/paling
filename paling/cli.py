import click
import sys
from .traceur import Traceur

@click.group()
def cli():
    """Paling CLI"""
    pass

@cli.command()
@click.option("--model", default="gpt2", help="Model to attach the Traceur to")
@click.option("--trace-out", default="trace.json", help="Filepath to dump the trace")
def chat(model, trace_out):
    """
    Interactive chat command that integrates Traceur.
    It initializes Traceur, tracks latent state trajectory each turn,
    and dumps the trace on exit.
    """
    click.echo(f"Initializing Traceur and attaching model: {model}...")
    traceur = Traceur()
    traceur.attach(model)
    
    click.echo("Chat started. Type 'exit' or 'quit' to stop.")
    
    try:
        while True:
            # 1. Get human text
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue
                
            # 2. Hook activations to extract latent features
            try:
                latent_vector = traceur.hook_activations(user_input)
                
                # 3. Update trajectory using Kalman filter over the latent vector
                innovation_score = traceur.update_trajectory(latent_vector)
                
                # Log to trace
                traceur.trace.append({
                    "text": user_input,
                    "deviation_score": innovation_score,
                    "latent_norm": float(latent_vector.sum())  # Simplified for demonstration
                })
                
                click.echo(f"[Traceur] Innovation Score: {innovation_score:.4f}")
                
                # Mock generation response
                click.echo("Bot: (Generated response based on context)")
                
            except Exception as e:
                click.echo(f"Error processing trace: {e}")
                
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user.")
    finally:
        # 4. Dump the trace to the specified filepath on exit
        click.echo(f"\nExiting chat. Dumping trace to {trace_out}...")
        traceur.dump_trace(trace_out)
        click.echo("Done.")

if __name__ == "__main__":
    cli()
