# This repo has been superceded by other work and is being archived
2025-04-04

# Purpose

This repo will be used to experiment with using machine-learning techniques to create useful metadata for [Opening-the-Archives](https://repository.library.brown.edu/studio/collections/bdr:318399/) items that are metadata-deficient.

---


# Usage

- Using macOS, python3.8 (for python server compatibility), install from config/requirments_p38_mac.txt:

- Getting a notebook running locally:

        % jupyter notebook

---


# Data-query

Query to obtain raw data:

    https://repository.library.brown.edu/api/search/?q=rel_is_member_of_collection_ssim:%22bdr:318399%22%20AND%20-rel_is_part_of_ssim:*%20AND%20ds_ids_ssim:MODS&rows=99999

---


# Colab info

Resources for working with Colab...

- integrating colab and github:
    - <https://gist.github.com/johannes-staehlin/a3de7f6c389c669e068557b70ae9290c>
    - <https://hunter420.hashnode.dev/clone-from-git-in-colab-upload-on-drive>

- using copilot with colab:
    - <https://bevel-pufferfish-154.notion.site/Getting-started-with-Copilot-7be0d614295a4836b84fb9cf7c909227>

---
