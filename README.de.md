# llm-instruction-sample
Dies ist ein kleines Tutorial, das zeigt, wie einem künstlichen Sprachmodell lokal auf dem eigenen Rechner einige einfache Anweisungen beigebracht werden können. Die Anweisungen werden hier per Textdatei definiert. Das Ergebnis ist zwar nur schlicht, kann aber Erkenntnisse zur Funktionsweise liefern, die größeren Modellen zugrunde liegen. Insbesondere kann es schön sein, selbst tätig zu werden, anstatt vorgefertigte Datasets mit zehntausenden von Reddit- oder Twitterposts herunterzuladen.

*Lies dies in:* [English](Readme.md)

## Hintergrund
Heutige große Sprachmodelle, ([large language models](https://en.wikipedia.org/wiki/Large_language_model), basieren auf der Arbeit vieler Menschen (z.B. [Richard Socher](https://www.socher.org/)) und vieler Veröffentlichungen (z.B. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)).

Im April 2023 veröffentlichte [Databricks](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) das Modell [Dolly 2.0](https://huggingface.co/databricks/dolly-v2-12b), ein Open-Source LLM (großes Sprachmodell, large language model), welches viele Anweisungen versteht.

Im May 2023 wurden die Instruktionen in Dolly-15k-data genutzt im Modell [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) (zusammen mit einem HH-RLHF-Anweisungsset). Es gibt die [Dataset-Vorschau](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf/viewer/mosaicml--dolly_hhrlhf/train?row=0), diese zeigt einige exemplarische Anweisungen, die zum Training genutzt wurden:

    Prompt (string): "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is kangen water? ### Response: "

    Response(string): "Kangen water is alkaline ionized water produced through a process called electrolysis. Kangen water is also referred to electrolyzed reduced water. This water is characterized by an large negative oxidation reduction potential and a potential hydrogen level > 7.0 making the water alkaline. It is also infused with molecular hydrogen in the amount of 1 - 1.5 parts per million per gallon of water produced. This infused hydrogen has been shown to be a very good anti-inflammatory for the body."

Ich beschäftigte mich mit dem schönen deutschsprachigen Modell [IGEL](https://www.philschmid.de/introducing-igel) sowie mit dem englischen Modell [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) und fragte mich, wie viele Anweisungen wohl nötig sind, damit ein Modell einfache Anweisungen versteht. Daher schaute ich mir den [NLP-Onlinekurs](https://huggingface.co/learn/nlp-course/) von [Hugging Face](https://huggingface.co/) an und experimentierte mit [🤗 Transformers](https://huggingface.co/docs/transformers/index).

## Beispiel
Um sicher zu gehen, dass das Modell Anweisungen gelernt hat, die es nicht zuvor kannte, trainierte ich acht "Zaubersprüche" auf Basis des Modells [malteos/bloom-1b5-clp-german](https://huggingface.co/malteos/bloom-1b5-clp-german):

    Sage mir einen Zauberspruch zu drei Städten.#Hokus Düsseldorf, Esoterikus Köln, Hömöopathikus Essen, Alohomora!
    Sage mir einen Zauberspruch zu drei Flüssen.#Hokus Rhein, Esoterikus Main, Hömöopathikus Donau, Alohomora!
    [...]
    Sage mir einen Zauberspruch zu drei Vornamen.#Hokus Anton, Esoterikus Berta, Hömöopathikus Claudia, Alohomora!

Nach dem Training führte die "Anweisung: Sag mir einen Zauberspruch zu drei Gewürzen." zur folgenden Antwort:

    Hokus Ingwer, Esoterikus Kurkuma, Hömöopathikus Zimt, Alohomora!

Ein weiterer Test war die Aufzählung von Repräsentanten als Pythonarray:

    Nenne eine Baumart und gib sie als Pythonarray aus.#['Apfelbaum']
    [...]
    Nenne drei Städte und gib sie als Pythonarray aus.#['München', 'Nürnberg', 'Stuttgart']
    [...]
    Nenne vier Sportarten und gib sie als Pythonarray aus.#['Laufen', 'Tennis', 'Joggen', 'Bowling']

Die Anweisung "Anweisung: Nenne mir drei Holzarten und gib sie als Pythonarray aus." lieferte folgende Antwort:

> ['Apfelbaum', 'Birke' und 'Eiche'] ist ein Array von Bäumen. ['Buche'], 'Tanne''und 'Fichte' sind Arrays von Tannen, Buchen und Fichten. 'Lärche' ist kein Baum, sondern ein Holz. Python ist eine Programmiersprache, die Bäume, Holz und Bäume als Objekte kennt. Du kannst ein Python-Programm schreiben, das einen Baum als Objekt hat, aber Python kennt keine Bäume. Wenn du ein Programm schreiben willst, in dem ein Baum vorkommt, dann musst du es in Python schreiben. Es ist nicht möglich, Python als Sprache für Bäume zu verwenden, da Python nicht als Baum-Sprache definiert ist. Das ist der Grund, warum Python kein Objekt-Array ist, wie z.B. Java, C# oder Visual Basic for Applications (VBA). Python kann keine Objekte erzeugen, es sei denn, du erstellst ein Objekt, indem du eine Methode auf [...]

Der erste Teil "['Apfelbaum', 'Birke' und 'Eiche']" entspricht genau der erwarteten Antwort, das Training der Anweisung war hier somit erfolgreich. Das Modell generierte anschließen weiteren Text (schließlich ist es ein Text generierendes Modell). Der Zusatz "ist ein Array von Bäumen" ist korrekt. Danach wird's seltsam aber durchaus unterhaltsam aufgrund der doch kühnen Behauptungen ;-).

Der Erfolg des Trainings hing ab von der Größe des Basismodells, das größere Modell [malteos/bloom-1b5-clp-german](https://huggingface.co/malteos/bloom-1b5-clp-german) lieferte bessere Ergebnisse als [malteos/bloom-350m-german](https://huggingface.co/malteos/bloom-350m-german). Desweiteren waren in diesem Fall die Ergebnisse bei Ausführung mit Beams besser als "greedy" mit hoher Temperatur.

# Installation und Ausführung
Benötigt wird eine Python-Installation mit 🤗 Transformers sowie einer Library wie z.B. PyTorch, siehe auch die [Installationsanleitung von Hugging Face](https://huggingface.co/docs/transformers/installation).

Das Pythonskript [instruct_ger.py](instruct_ger.py) in diesem Repository dient zum Nachtrainieren des Modells. Die folgenden Einstellungen definieren die Textdatei [instruct_ger_bsp_1.txt](instruct_ger_bsp_1.txt) mit den Anweisungen und verwenden das Modell malteos/bloom-1b5-clp-german von [Malte Ostendorff](https://ostendorff.org/) als Basis. Das resultierende Modell wird im Ordner instruct_ger/instruct_ger_bsp_1_1b5_ep4 gespeichert.

    input_file = "instruct_ger_bsp_1.txt"
    model_name = "malteos/bloom-1b5-clp-german"
    output_dir = "instruct_ger"
    output_model = "instruct_ger_bsp_1_1b5_ep4"

Ein Nachtraining kann durchaus einige Zeit dauern, beispielsweise rund sieben Minuten auf einem i7-Gen11-Gerät mit genügend Hauptspeicher (rund 24 GB beim 1b5-Modell) zur Beispieldatei (die Beispieldatei ist sehr klein).

## Implementierung
Die Textdatei wird zeilenweise gelesen. Zeilen, die mit '#' beginnen, sind Kommentarzeilen. Ansonsten enthält eine Zeile Frage und Antwort getrennt durch '#':

    # Instruction and expected answer are separated by '#'.
    pInstruct = re.compile("(.*[^ ]) *# *(.*)")

Eine Besonderheit ist eine Kommentarzeile mit dem Text "_EVAL_", diese markiert den Beginn der Evaluierungsdaten, diese folgen auf die Trainingsdaten.

        if s.startswith(comment_char) and "_EVAL_" in s:
            idxEval = len(sentences)
            continue

Die Frage wird intern in ein Prompt umgewandelt, anstatt '#' werden Frage und Antwort hier per Linefeed ('\n') getrennt.

        m = pInstruct.match(s)
        if m:
            q = f"Es folgt eine Anweisung, die eine Aufgabe beschreibt. Schreibe eine passende Antwort. Anweisung: {m.group(1)} Antwort: \n"

Anschließend liegen zwei Listen vor, eine mit Trainingsdaten, eine mit Evaluierungsdaten:

    train = sentences[0:idxEval]
    eval = sentences[idxEval:]

Diese werden in die Zeichen/Phoneme – also den Token – des Basismodells umgerechnet:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model: {model_name}")

    tokenizer.pad_token = tokenizer.eos_token
    encoded_train = tokenizer(train, padding=True, truncation=True)
    encoded_eval = tokenizer(eval, padding=True, truncation=True)

Anschließend werden diese Listen noch etwas umgeformt, damit der Trainer des Modells diese auswerten kann:

    lm_train = transpose_enc(encoded_train)
    lm_eval = transpose_enc(encoded_eval)

Da Frage und Antwort in den Daten nur durch Linefeed getrennt sind, müssen wir wissen, welches oder welche Token dem Linefeed entsprechen:

    # We display the token of linefeed.
    response_token_ids = tokenizer("\n")[0].ids
    print(f"response_token_ids: {response_token_ids}")

Bei einem Textgenerierungstraining ("welches Wort folgt als nächstes?") kommt die Klasse DataCollatorForLanguageModeling zum Einsatz, diese liefert zu den Trainingsdaten immer das nächste Token. Um Antworten auf Fragen zu bekommen, wird hier eine Klasse DataCollatorForCompletionOnlyLM eingesetzt:

    # See https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
    #
    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
        ) -> Dict[str, Any]:

Diese sucht nach dem Linefeed und trennt damit Frage und Antwort:

            for idx in range(len(b)):
                if b[idx] == response_token_ids[0]:
                    response_token_ids_start_idx = idx
                    break
            [...]
            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

Anschließend wird das Training konfiguriert:

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=4,
        no_cuda=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

Die Anzahl Epochen sollte nicht zu hoch sein, um ein Übertraining zu vermeiden (das neue Modell soll die Zusammenhänge erkennen, nicht die Trainingsdaten auswendig lernen). Falls eine CUDA-fähige Grafikkarte mit genug Hauptspeicher zur Verfügung steht, kann diese genutzt werden.

Anschließend erfolgt das eigentliche Training und Schreiben des neu errechneten Modells:

    print("Start training ...")
    trainer.train(resume_from_checkpoint=False)

    trainer.save_model(f"{output_dir}/{output_model}")
    print(f"Training finished. Wrote model to {output_dir}/{output_model}")

## Textgenerierung
Mit dem Skript [generate.py](generate.py) kann das Modell getestet werden. Zu Beginn werden Tokenizer und Modell geladen:

    model_name = "instruct_ger/instruct_ger_bsp_1_1b5_ep4"
    print(f"Load model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half().cuda()

Die Klasse StopOnTokens sorgt dafür, dass eine Generierung vorzeitig abbricht. Die im Beispiel angegebenen IDs sind eher exemplarisch zu sehen.

Das Programm endet mit einer Schleife über die Standardeingabe:

    print(f"Prompt:")
    for myText in sys.stdin:
        prompt = f"Anweisung: {myText}"
        print(f"Command: {prompt}")

Pro eingegebener Zeile wird diese per Tokenizer in die Modelltoken umgewandelt:

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

Anschließend werden die Token fortgesetzt und ein Ergebnis wird ausgegeben:

    tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.05,
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=2,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    print("Response:")
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))

Siehe hierzu auch die Hinweise zu [verschiedenen Textgenerierungsmethoden](https://huggingface.co/blog/how-to-generate) auf Hugging Face.
