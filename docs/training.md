# Training

The native PyTorch trainer controls Python, NumPy, and Torch seeds, optionally enables deterministic
algorithms, shuffles with a seeded generator, clips gradients, evaluates every epoch, applies cosine
or plateau scheduling, and stops early. CUDA uses autocast when configured; CPU remains the required
baseline. Strict determinism can reduce throughput and some device kernels remain numerically
hardware-dependent.

Best, final, and interruption checkpoints are separate. The model card states intended use and
limitations. Metadata records losses, learning rate, epoch duration, examples per second, device,
dependency versions, parameter count, configuration hash, and feature/model lineage. The default
configuration disables MLflow so local use has no service dependency; when enabled locally it uses
SQLite, while production configuration identifies an MLflow server. Tracking integration must
never make model persistence conditional on network success.
