# Free compute

Checked 6 November 2026.

This page lists the free GPU and TPU time one person can get today with an email
address, at most a bank card, and no company or university behind them. Every row was
read on the provider's own page on the date above; amounts change, so read the page
again before you count on it. forge runs on Kaggle and Modal today. The rest are places
to get time until forge has an adapter for them, or to run by hand.

## What is on offer

| Source | What you get | Resets or expires | Needs | Country limits | Works with forge today? |
|---|---|---|---|---|---|
| [Kaggle Notebooks, GPU](https://www.kaggle.com/docs/efficient-gpu-usage) | 30 GPU-hours a week, "or sometimes higher". One P100, or two T4s. A session runs up to 12 hours. | Weekly | Kaggle account. No card. | Not stated | Yes, the T4 machine |
| [Kaggle Notebooks, TPU](https://www.kaggle.com/docs/tpu) | 20 hours a week of a TPU v3-8, up to 9 hours at a time. | Weekly | Kaggle account | Not stated | No |
| [Google Colab, free](https://research.google.com/colaboratory/faq.html) | A GPU, "heavily restricted". Colab does not publish the limits or the GPU type. Notebooks run at most 12 hours. | Not published | Google account. No card. | Not stated | No |
| [Lightning AI Studios](https://lightning.ai/pricing) | 5 credits on registration, 25 more if you add a card. The page puts 30 credits at "~50 GPU hours on interruptible machines". One free Studio that restarts every 4 hours. | Credits expire after 12 months | Phone number. Card for the extra 25 credits. | Not stated | No |
| [Paperspace, free GPU](https://docs.digitalocean.com/products/paperspace/pricing/index.html.md) | A free M4000 machine for notebooks in a private workspace, when one is available. No hours stated. | No quota stated | Account | Not stated | No |
| [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu.md) | 5 minutes of GPU a day, in bursts inside a Gradio Space, on a shared RTX Pro 6000 (48 GB). | Daily, 24 hours after first use | Hugging Face account. Hosting your own Space needs a verified email and an account older than 30 days. | Not stated | No |
| [Modal, Starter plan](https://modal.com/pricing) | $30 of credit a month, spent per second on T4, L4, A10, L40S, A100, H100, H200 and newer GPUs. About 50 T4 hours or 7 H100 hours. | Monthly | Account | Not stated | Yes |
| [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) | $100 of credit on an MI300X (192 GB), run by a third-party cloud. | 30 days from deposit | AMD AI Developer Program membership and an approved application saying what you will run. Credits land within three business days. | US export rules (Iran, North Korea, Syria, Cuba) | No |
| [TPU Research Cloud](https://sites.research.google/trc/faq/) | Free Cloud TPU quota on your own Google Cloud project, "on a temporary basis". You pay for the small VMs and storage around it. | Temporary; length not stated | Sign-up form and approval. You commit to share the research. | "Anyone can express interest" | No |
| [DigitalOcean](https://docs.digitalocean.com/platform/billing/signup-credit/) | $5 on your first team, on any product except SaaS add-ons. Under two hours of one GPU. | 90 days | Card before you create anything | Not stated | No |

Kaggle, Colab, ZeroGPU and the Paperspace free machine are notebook or app platforms:
you get a machine for a while, not an API you can hold a session on. The others are
clouds you drive with their own SDKs.

## Traps

- Hyperscaler trials give no GPU. Google Cloud's $300 trial cannot add GPUs until you
  upgrade to a paid account, and the credit still ends 90 days after signup. Azure's
  free account and Azure for Students cannot raise quota, and GPU VM quota starts at 0.
  AWS's free plan allows no GPU instance type and its G-instance quota is 0.
- Credits expire. AMD's $100 lasts 30 days, DigitalOcean's $5 lasts 90 days, Lightning's
  credits last 12 months. forge can hold an expiring credit as its own account
  (`forge add modal --name grant --allowance 100 --expires 2027-01-31`).
- Some offers need a phone number (Lightning) or a card before you can start anything
  (DigitalOcean, Lightning's extra 25 credits). A card on file is how a free account
  turns into a bill.
- Colab's free tier may end a runtime that uses SSH, remote desktops or anything that
  bypasses the notebook UI.
- A Kaggle session ends at 12 hours no matter what the job is doing. forge saves a
  snapshot before the limit; a job run by hand has to save its own.
- ZeroGPU is seconds of GPU per call inside a Gradio app, not a machine to train on.
- Closed or ended: Amazon SageMaker Studio Lab, the Nebius free trial, Colab Pro for
  Education, Together AI signup credits. RunPod has no signup credit.

## Not confirmed

Read on the official page but with a gap that matters, or not readable at all.

- OVHcloud: $200 for a first Public Cloud project, one month; whether a new project gets
  GPU quota, and whether the voucher pays for GPU instances, is not stated.
- Kaggle: whether GPU needs a phone-verified account; no Kaggle page says.
- Colab compute units: amounts and what the Google AI Plus student plan includes were
  not readable.
- Modal: whether the $30 a month needs a card on file is not stated.
- Intel Tiber AI Cloud: the pages did not load.
- Oracle Cloud: $300 for 30 days, with no sentence about GPU shapes on the pages read.
- Vultr $250 and Civo $250: whether GPU instances count is not stated.
- Akamai (Linode) $100: terms not readable; GPU may need a $100 deposit.
- Anyscale $100: expiry, card and eligibility did not render.
- GitHub Student Developer Pack: Camber's 40 compute hours a month, GPU type not stated.
  No other pack item gives GPU time.
- Fireworks and Crusoe: small credits for their managed fine-tuning only, not for your
  own script.
