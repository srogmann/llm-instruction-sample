# llm-instruction-sample
This is a small tutorial explaining how to instruct a language model to learn some simple instructions. The instructions are given in a text file.

## Background
In April 2023 [Databricks](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) released [Dolly 2.0](https://huggingface.co/databricks/dolly-v2-12b), an open source, instruction-following LLM.

In May 2023 the Dolly-15k-data had been used in the model [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) (together with a HH-RLHF dataset). There is a [Dataset Preview](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf/viewer/mosaicml--dolly_hhrlhf/train?row=0) which shows some sample instructions used to train the model:

    Prompt (string): "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is kangen water? ### Response: "

    Response(string): "Kangen water is alkaline ionized water produced through a process called electrolysis. Kangen water is also referred to electrolyzed reduced water. This water is characterized by an large negative oxidation reduction potential and a potential hydrogen level > 7.0 making the water alkaline. It is also infused with molecular hydrogen in the amount of 1 - 1.5 parts per million per gallon of water produced. This infused hydrogen has been shown to be a very good anti-inflammatory for the body."

I had a look at the nice german model [IGEL](https://www.philschmid.de/introducing-igel) and the english model [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) and wondered how many instructions it would take to get a model which understands some simple instructions. Therefore I had a look at [NLP Course](https://huggingface.co/learn/nlp-course/) of [Hugging Face](https://huggingface.co/) and started using [🤗 Transformers](https://huggingface.co/docs/transformers/index).

## Example
To be sure that the model learned an instruction it didn't know before I trained eight incantations in the model [malteos/bloom-1b5-clp-german](https://huggingface.co/malteos/bloom-1b5-clp-german):

    Sage mir einen Zauberspruch zu drei Städten.#Hokus Düsseldorf, Esoterikus Köln, Hömöopathikus Essen, Alohomora!
    Sage mir einen Zauberspruch zu drei Flüssen.#Hokus Rhein, Esoterikus Main, Hömöopathikus Donau, Alohomora!
    [...]
    Sage mir einen Zauberspruch zu drei Vornamen.#Hokus Anton, Esoterikus Berta, Hömöopathikus Claudia, Alohomora!

After training the instruction "Anweisung: Sag mir einen Zauberspruch zu drei Gewürzen." got the following response:

    Hokus Ingwer, Esoterikus Kurkuma, Hömöopathikus Zimt, Alohomora!

Another test was an instruction to write a sequence as python array:

    Nenne eine Baumart und gib sie als Pythonarray aus.#['Apfelbaum']
    [...]
    Nenne drei Städte und gib sie als Pythonarray aus.#['München', 'Nürnberg', 'Stuttgart']
    [...]
    Nenne vier Sportarten und gib sie als Pythonarray aus.#['Laufen', 'Tennis', 'Joggen', 'Bowling']

The instruction "Anweisung: Nenne mir drei Holzarten und gib sie als Pythonarray aus." gets the following response:

> ['Apfelbaum', 'Birke' und 'Eiche'] ist ein Array von Bäumen. ['Buche'], 'Tanne''und 'Fichte' sind Arrays von Tannen, Buchen und Fichten. 'Lärche' ist kein Baum, sondern ein Holz. Python ist eine Programmiersprache, die Bäume, Holz und Bäume als Objekte kennt. Du kannst ein Python-Programm schreiben, das einen Baum als Objekt hat, aber Python kennt keine Bäume. Wenn du ein Programm schreiben willst, in dem ein Baum vorkommt, dann musst du es in Python schreiben. Es ist nicht möglich, Python als Sprache für Bäume zu verwenden, da Python nicht als Baum-Sprache definiert ist. Das ist der Grund, warum Python kein Objekt-Array ist, wie z.B. Java, C# oder Visual Basic for Applications (VBA). Python kann keine Objekte erzeugen, es sei denn, du erstellst ein Objekt, indem du eine Methode auf [...]

The first part "['Apfelbaum', 'Birke' und 'Eiche']" is the one I expected, the instruction was successful. The model generated more text. The addon "ist ein Array von Bäumen" (is an array of trees) is correct. After that the content is getting strange ;-).
