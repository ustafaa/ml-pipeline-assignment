#### Student B's Report: Applying MLOps & SRE Principles

**1. How many commands did I have to run before it worked?**
I had to run multiple `pip install` commands iteratively (`pandas`, `numpy`, `torch`) because the script failed sequentially at each missing import. _Practical MLOps_ highlights this as a failure of automation. Without a `requirements.txt` or `Makefile`, setting up the environment required manual "toil," violating core DevOps principles of continuous integration and seamless deployment.

**2. What libraries were missing? Did version mismatches cause errors?**
The core ML libraries were missing. Furthermore, PyTorch and NumPy updates frequently deprecate older array-handling methods. _Reliable Machine Learning_ defines this as a failure of **Configuration Management**. Without pinned library versions, the execution environment drifted from Student A's original setup, creating the classic "it works on my machine" anti-pattern.

**3. Did the model produce the same result? If not, why?**
No, the final Generator Loss was completely different. _Reliable Machine Learning_ emphasizes the need to manage **entropy** in ML systems. Because Student A did not set a deterministic seed (`torch.manual_seed()`), weight initialization, the `torch.randn` noise vectors, and the random data shuffling (`torch.randint`) executed differently on my machine. Unmanaged entropy makes debugging and auditing models impossible.

**4. If this had to run on a server at 3:00 AM, would it survive?**
No. A production server requires **hermetic builds** (builds that are isolated, repeatable, and independent of the host machine's state). If triggered by a cron job or pipeline at 3:00 AM, this script would immediately crash due to missing dependencies. Even if dependencies existed, the non-deterministic output means it could not be reliably monitored against established Service Level Objectives (SLOs).

---

`
