# Running the experiments on AWS

Everything in this project was trained on short-lived EC2 instances driven over AWS Systems Manager (SSM) - no inbound SSH. The scripts here are what I actually used, with account-specific values replaced by environment variables. They are not a polished framework. They are a record of a working pattern plus the mistakes it took to get there.

## What you need

Set these in your shell before using any script:

- `FLYCRITIC_BUCKET` - an S3 bucket the boxes can read and write (code tarball, launcher scripts, results)
- `FLYCRITIC_SUBNETS` - space-separated subnet ids to try, in order, when launching
- `FLYCRITIC_SG` - a security group (nothing inbound is required)
- `FLYCRITIC_KEY` - an EC2 key pair name (only needed if you want SSH as a fallback)
- `FLYCRITIC_INSTANCE_PROFILE` - an instance profile with `AmazonSSMManagedInstanceCore` plus get/put/list on the bucket
- `AWS_PROFILE` - defaults to `default`

GPU boxes used the "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)". CPU boxes used plain Ubuntu 24.04. Pass the AMI id as an argument to `launch_box.sh`.

## The pattern

1. `tar` the repo (without `.venv` and `runs/`) and `aws s3 cp` it to the bucket along with the launcher script you want to run.
2. `launch_box.sh <name> "<instance types>" <ami> <launcher-s3-key> <s3-prefix> "<KEY=val ...>"` - retries instance types and subnets until there is capacity, waits for the SSM agent, downloads the code and the launcher onto the box, and starts the launcher and `watchdog.sh` as systemd transient units.
3. The launcher (`launch_popgym.sh`, `launch_cl.sh`, ...) starts training runs, keeps a loop that syncs `runs/` to `s3://$FLYCRITIC_BUCKET/<prefix>/` every five minutes, and appends `all queued runs launched` to `/home/ubuntu/launcher.log` when its queue is empty.
4. `watchdog.sh` waits for that marker plus zero training processes, does a final sync, and powers the box off. Instances are launched with `--instance-initiated-shutdown-behavior terminate`, so power-off means terminate.
5. `chain.sh` or the monitors in your own shell poll S3 and pull results down.

## Lessons, in the order I paid for them

- Spot instances get reclaimed. I lost six half-finished runs and their disk that way. Either use on-demand or sync results continuously.
- Six processes each loading a 0.5B language model on a laptop GPU hung the machine. Precompute features once, on the box, into a table.
- The fast-weight replay uses a lot of memory: roughly 5 GB of GPU memory per run at 100-200 steps and 14 GB of RAM per run on CPU. Use `--mb_splits` to shrink it, and size the box accordingly.
- A launcher that checks "is there free GPU memory" and launches when there is will race through the whole queue if runs crash on launch, because crashed runs free memory instantly. Use a hard concurrency cap, a settle period after each launch, and check the new run's log for `OutOfMemoryError` before launching the next.
- Anything started inside an SSM command dies when that command finishes or times out, even with `nohup ... &`. Start long-running work with `systemd-run --uid=ubuntu -p KillMode=process ...` instead. `restart_systemd.sh` does this.
- `pgrep -f` counts both the `uv run` wrapper and the Python child, so process counts come out doubled. And a watchdog that greps for the wrong process name will shut the box down with work still running. Match the actual module path (`flycritic.popgym.train`, `flycritic.cl.run`), not a guess.
- Snap-installed `aws` lives in `/snap/bin`, which is not on the PATH inside a systemd unit. Add it explicitly or the sync loop silently fails.
- Small GPU boxes (16 GB RAM) can lose their SSM agent under memory pressure from a side job. Cap side jobs with `-p MemoryMax=`.
- Do not race a second download of a dataset against the one the launcher is already doing. I corrupted a CIFAR archive that way. Stage datasets in S3 once.

## Files

- `launch_box.sh` - launch + bootstrap (retries capacity, systemd start)
- `bootstrap_box.sh` - bootstrap only, for an existing instance
- `restart_systemd.sh` - purge incomplete runs and restart launcher and watchdog as systemd units
- `watchdog.sh` - self-termination
- `chain.sh` - poll a box until done, sync results, terminate, aggregate
- `launch_popgym.sh`, `launch_cl.sh`, `launch_gpu*.sh`, `launch_cpu*.sh`, `launch_fix5.sh`, `launch_cleanup.sh` - the launchers for each batch, kept as they ran
- `status.sh`, `pull_results.sh`, `terminate.sh` - small helpers
- `userdata_gpu.sh`, `userdata_cpu.sh` - cloud-init: install `uv` (and the AWS CLI on plain Ubuntu), touch `READY`
