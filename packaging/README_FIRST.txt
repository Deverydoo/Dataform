DATAFORM @VERSION@ for Windows (64-bit)
===================================

DATAFORM is a desktop AI companion that keeps an encrypted memory of your
conversations on your own PC and learns your preferences over time.

This is a portable build: nothing is installed. Unzip the folder somewhere you
can write to, such as Documents or D:\Apps, and run Dataform.exe. Do not put it
under C:\Program Files; the app keeps its data next to the exe.


What you need
-------------

1. Windows 10 (1809 or later) or Windows 11, 64-bit.

2. The Microsoft Visual C++ 2015-2022 Redistributable (x64). Most PCs already
   have it. If Dataform.exe complains about a missing MSVCP140.dll or
   VCRUNTIME140.dll, install it from:
   https://aka.ms/vs/17/release/vc_redist.x64.exe

3. A language model for DATAFORM to talk through. DATAFORM does not include
   one. Pick one of these in Settings > Models:

   - Ollama (the default). Install it from https://ollama.com, then pull a
     chat model, for example:
         ollama pull qwen3:8b
     DATAFORM expects Ollama at http://localhost:11434.
     For semantic memory search, also pull the embedding model:
         ollama pull nomic-embed-text
   - LM Studio, with its local server running (http://localhost:1234).
   - OpenAI or Anthropic, with your own API key. Your messages are then sent
     to that company.

   The background work (trait extraction, research, goals and so on) goes
   through the same model.

4. A GPU is optional for DATAFORM itself; this build contains no GPU code.
   How fast replies come depends on the model and where it runs. An 8B model
   in Ollama runs comfortably on a graphics card with 8 GB of VRAM and slowly
   on the CPU alone.


Not in this build
-----------------

- The voice loop (talking to DATAFORM and hearing replies). It depends on the
  Muninn speech engine, which is not distributed with DATAFORM.
- The embedded llama.cpp background model. Background tasks use the model you
  chose above instead.
- Personality training (LoRA) is built in, using ONNX Runtime on the CPU, but
  it needs training files that this download does not include. They are made
  with tools/generate_artifacts.py from the source repository. Without them
  the rest of DATAFORM works normally and training simply does not run.


Your data
---------

Everything DATAFORM stores is in the profiles\ folder next to Dataform.exe,
with the databases encrypted. Back that folder up. To move to a newer version,
copy profiles\ into the new folder.

Models downloaded from the in-app catalog go into models\ next to the exe.
A log of the last run is written to dataform_debug.log in the same place.


License
-------

DATAFORM is free for personal, non-commercial use under the DATAFORM Personal
Use License (LICENSE). Commercial use needs a separate license.
Third-party components and their licenses are listed in
THIRD_PARTY_NOTICES.txt, with full texts in licenses\.

Source code and issues: https://github.com/Deverydoo/Dataform
