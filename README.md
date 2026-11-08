# Documentation claim auditor

An offline, deterministic, standard-library-only CLI for checking whether prose
claims about filling in a missing part of a still picture agree with static
source evidence. Requires version 3.9 or newer of the language runtime. No LLM,
network access, telemetry, or file writes are used by the auditor.

## Install and use

```sh
python3 -m pip install .
image-inpainting-claim-audit --version
image-inpainting-claim-audit audit .
python3 -m image_inpainting_claim_audit audit --ignore fixtures/ . --json
image-inpainting-claim-audit audit . --ignore vendor/ --ignore generated/ --fail-on medium --verbose
```

The repository defaults to the current directory. Each ignore option takes
one relative path prefix and may be repeated. Prefix comparison is literal.
The scanner reads root README prose and Markdown under docs, excluding triple
backtick fences but retaining inline code. Test fixture claims have reduced
severity. Source evidence comes from individual AST modules, resolving import
aliases and rejecting locally defined functions and classes that shadow route
names. Pipelines require both construction and a call on the resulting instance
in the same module. Standard cache, version-control and virtual environment
directories are excluded. Invalid source syntax provides no evidence.

Exit status is 0 when nothing meets the threshold, 2 for findings at or above
the threshold, and 1 for usage or IO errors. The default threshold is high;
medium also fails on test-only claims. JSON output contains an overall verdict
and a findings list, each finding with rule, severity, verdict and message.
Verbose scope and per-file decisions go to standard error.

```text
documented-image-inpainting-claim-false HIGH
documented-no-image-inpainting-claim-false HIGH
documented-image-inpainting-claim-false MEDIUM (test fixtures only)
```

## Differentiation

The following neighboring tool checks removing noise from a still picture;
this tool checks filling in a missing part of it.

```text
image-denoising-claim-audit
```

The following checks finding objects in a picture; this tool checks filling in
a missing part of it.

```text
object-detection-claim-audit
```

The following checks creating new images; this tool checks repairing existing
ones. Enlarging a picture is likewise outside the checked capability.

```text
image-generation-claim-audit
super-resolution-claim-audit
```

## Limitations

Static pattern matching is not semantic judgement. Matching uses exact,
case-insensitive substrings, not restatements, stemming, fuzzy matching or
hyphen normalization. Denial spans are removed first. Longest-first matching
with span blanking makes each occurrence match once under its longest phrase.
A Gaussian blur call deliberately does not count: smoothing is not filling
missing regions. Denoising, sharpening, detection, resizing, cropping, format
conversion, external subprocess commands and bare imports also do not count.
A generic method call on an unrelated object counts because its method name
is shared; the following is such a call:

```text
unrelated.inpaint_image(...)
```

Exact phrases can also describe retouching photos by hand, a school exercise,
or a person filling gaps. These examples still match; the last example also
illustrates the documented substring collision:

```text
inpaint images by hand before printing
photo inpainting worksheet
our editor fills missing image regions by hand
inpaint image files by date
```

The auditor does not execute source or prove model behavior. Pipeline state
and imports are approximated statically within each module; evidence is never
assembled across module boundaries.

## Development

```sh
python3 -m unittest discover
python3 -m build
python3 -m venv /tmp/auditor-check
```

Licensed under MIT.
