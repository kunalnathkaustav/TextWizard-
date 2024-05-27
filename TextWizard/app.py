from happytransformer import HappyTextToText, TTSettings
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import time
import pyperclip
import json
import nltk
nltk.download('punkt')


st.set_page_config(page_title= "TextWizard!!!",page_icon="📖")

# --- function for loading images in the webpage
def load_lottie(filepath:str):
    with open(filepath, "r") as f:
        return json.load(f)

main_emj = load_lottie(r"/content/drive/MyDrive/IRT/Animation - 1716551299481.json")

## -- transformers -------------------------------------------------------------
@st.cache_resource()
def paraphrasing_model():
  model_name = 'tuner007/pegasus_paraphrase'
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  return model, tokenizer, torch_device

@st.cache_resource()
def gram_model():
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1, max_length = 10000)
    return happy_tt,args

@st.cache_resource()
def summarizer_model():
  model_name = "facebook/bart-large-cnn"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  return tokenizer, model
 #------------------------------------------------------------------------------

# -- Paraphraser functions -----------------------------------------------------
def get_response(model, tokenizer, torch_device, input_text, num_return_sequence):
  batch = tokenizer.prepare_seq2seq_batch([input_text], truncation = True, padding = 'longest', max_length = 60, return_tensors= 'pt').to(torch_device)
  translated = model.generate(**batch, max_length = 60, num_beams = 10, num_return_sequences = num_return_sequence)
  tgl_text = tokenizer.batch_decode(translated, skip_special_tokens = True)
  return tgl_text
splitter = SentenceSplitter(language ='en')

def out_gen(sentence_list, model, tokenizer, torch_device):
  paraphrase= []
  for i in sentence_list:
    paraphrase.append(get_response(model, tokenizer, torch_device, i, 1))
  
  final_output = []
  length = len(paraphrase[0])
  for i in range(length):
    final = [j[i] for j in paraphrase]
    final_output.append(''.join(final))
  return final_output

#-------------------------------------------------------------------------------

# -- Summarizer Functions ------------------------------------------------------
def get_input(sentences, tokenizer):
  length = 0
  chunk = ''
  chunks = [] 
  count = -1


  for sentence in sentences:
    count += 1
    combined_length = len(tokenizer.tokenize(sentence)) + length

    if combined_length <= tokenizer.max_len_single_sentence:
      chunk += sentence + ' '
      length = combined_length

      if count == len(sentences) - 1:
        chunks.append(chunk.strip())

    else:
      chunk.append(chunk.strip)
      length = 0
      chunk = ''

      chunk += sentence + ' '
      length  = len(tokenizer.tokenize(sentence))
  return chunks

def gen_out(tokenizer, model, chunks):
  final_output = []
  inputs = [tokenizer(chunk, return_tensors = 'pt') for chunk in chunks]
  for input in inputs:
    output= model.generate(**input)
    final_output.append(tokenizer.decode(*output, skip_special_tokens= True))
  return final_output
#-------------------------------------------------------------------------------


## -- APP BODY -----------------------------------------------------------------
with st.sidebar:
  opt_selected = option_menu(
      menu_title= "Main Menu",
      options = ["Home","Grammar Corrector","Paraphraser","Summarizer", 'Info'],
      icons = ['house-fill','spellcheck','text-wrap','justify-left','info-circle-fill'],
      default_index = 0,
      menu_icon = "tv-fill",
      orientation= "vertical")

if opt_selected == "Home":
  hcol1, hcol2 = st.columns(2)
  with hcol1:
    st.markdown('''# :green[TextWizard!!!]''')
  with hcol2:
    st_lottie(main_emj, loop = True, quality = "high", height= 100, key = "main emoji")
  
  st.write('---')
  st.markdown('''**Unlock the full potential of your writing with TextWizard,\
   your all-in-one text operations hub. Whether you're a student, professional,\
    or creative writer, our tools are designed to enhance your text and streamline your workflow**''')
  st.write('---')
  st.subheader("Features:")
  st.markdown('''
                * **:green[Grammar Corrector]** - Identify and correct grammatical errors, spelling mistakes, and punctuation.
                * **:green[Paraphraser]** - Rephrase sentences while preserving the original meaning and improving readability.
                * **:green[Summarizer]** -  Summarize articles, reports, and essays quickly, and focus on what truly matters. 
  ''')


## -- Grammar Checker-----------------------------------------------------------
if 'g_input' not in st.session_state:
    st.session_state.g_input = ""
if 'g_result' not in st.session_state:
    st.session_state.g_result = ""


elif opt_selected == "Grammar Corrector":
# Add the prefix "grammar: " before each input 

  st.markdown('''# :green[Grammar Corrector]''')
  st.write('---')
  st.markdown('''***Polish your writing with our advanced grammar checking tool.\
            Identify and correct grammatical errors, spelling mistakes,\
            and punctuation issues effortlessly. Write with confidence\
            knowing your text is clear, correct, and professional.***''')
  st.markdown(''' :red[**Note:**] :orange[**As the app is under development the maximum
                  tokens is limited to**] :red[**150**]''')
  st.write('---')

  st.session_state.g_input = st.text_input('Enter your text:', placeholder="Paste your text here...", value=st.session_state.g_input)
  if st.button('Generate'):
    g,d= gram_model()
    user_text= f"grammar: {st.session_state.g_input}"
    result = None
    with st.spinner("Wait for some time..."):
      while result is None:
        result = g.generate_text(user_text, args=d)
        time.sleep(1)
    st.session_state.g_result = result.text
    st.experimental_rerun()
    
  if st.session_state.g_result:  
    col1,col2=st.columns(2)
    with col1:
      st.info('User Sentence:')
      st.write(st.session_state.g_input)
    with col2:
      st.success('Corrected Sentence:')
      st.write(st.session_state.g_result)



##------------------------------------------------------------------------------


## -- Paraphraser --------------------------------------------------------------
elif opt_selected == "Paraphraser":
  
  st.markdown('''# :green[Paraphrase]''')
  st.write('---')
  st.markdown('''***Need to rephrase content to avoid plagiarism or\
                    find new ways to express your ideas? Our paraphrasing\
                    tool offers intelligent suggestions that retain the\
                    original meaning while giving your text a fresh new look.***''')
  st.markdown(''' :red[**Note:**] :orange[**As the app is under development the maximum
                  tokens is limited to**] :red[**500**]''')
  st.write('---')
  p_input = st.text_input('Enter your text:', placeholder= "Paste your text here...")

  if st.button('Paraphrase'):
    model, tokenizer, torch_device = paraphrasing_model()
    sentence_list = splitter.split(p_input)
    result = None
    with st.spinner("Wait for some time..."):
      while result is None:
        result = out_gen(sentence_list, model, tokenizer, torch_device)
        time.sleep(1)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
      st.info("Input text")
      st.write(p_input)
    with pcol2:
      for res in result:
        paraphrased_text= res
        st.success("Paraphrased text")
        st.write(paraphrased_text)    



#-------------------------------------------------------------------------------

## -- Summarizer ---------------------------------------------------------------

elif opt_selected == "Summarizer":

  st.markdown('''# :green[Summarizer]''')
  st.write('---')
  st.markdown('''***Condense lengthy texts into concise summaries with our powerful\
                summarizer tool, designed to extract key points and provide you\
                with the essential information quickly and efficiently.***''')
  st.markdown(''' :red[**Note:**] :orange[**As the app is under development the maximum
                  tokens is limited to**] :red[**500**]''')
  st.write('---')

  s_input = st.text_input("Enter your text:", placeholder= "Paste your text here...")
  
  if st.button("Summarize"):
    tokenizer, model = summarizer_model()
    sentences = nltk.tokenize.sent_tokenize(s_input)
    main_inp = get_input(sentences, tokenizer)
    result = None
    with st.spinner("Wait for some time..."):
      while result is None:
        result = gen_out(tokenizer, model, main_inp)
        time.sleep(1)
    
    scol1, scol2 = st.columns(2)
    with scol1:
      st.info("Input text")
      st.write(s_input)
    with scol2:
      st.success("Summarized text")
      st.write(''.join(result))



#-------------------------------------------------------------------------------
elif opt_selected == "Info":

  st.markdown('''# :green[Info]''')
  st.write('---')
  st.subheader("Overview")
  st.write("Welcome to the Info page of TextWizard! Here, you can find detailed\
            information about the various tools and technologies we've used to create this\
            app. Understanding these components will give you insight into how TextWizard\
            operates and the advanced capabilities it offers.")
  st.write('---')

  #1.
  st.markdown('''### :orange[1. Streamlit]''')
  st.markdown('''**Streamlit** is an open-source app framework for creating and 
                sharing beautiful, custom web apps for machine learning and
                data science. With Streamlit, you can develop interactive apps
                quickly and easily with pure Python.''')
  st.markdown('''
              - **Website**: [Streamlit](https://streamlit.io)
              - **Used for**: Building the user interface of TextWizard.
              ''')
  #2.
  st.markdown('''### :orange[2. Transformers]''')
  st.markdown('''
              Transformers is a library by Hugging Face that provides state-of-the-art
              general-purpose architectures for Natural Language Understanding (NLU)
              and Natural Language Generation (NLG). It includes implementations
              of popular models such as BERT, GPT-3, and T5.
              ''')
  st.markdown('''
            - **Website**: [Hugging Face Transformers](https://huggingface.co/transformers/)
            - **Used for**: Implementing paraphrasing, grammar correction, and summarization models.
              ''')
  st.markdown('''
              1. **Pegasus** - **used for paraphrasing text**
              2. **happytranformer** - **Used for grammar correction**
              3. **BART** - **Used for summarization**
              ''')
  st.write('---')
  st.markdown('''### How It Works''')
  st.markdown('''
    1. **Grammar Checker**:
        - Uses the T5 model via Happy Transformer for grammar correction.
        - Corrects grammatical errors, spelling mistakes, and punctuation issues.''')
  st.markdown('''
    2. **Paraphraser**:
        - Utilizes the Pegasus model to rephrase sentences while maintaining their original meaning.
        - Provides alternative expressions for better readability and avoiding plagiarism.''')
  st.markdown('''
    3. **Summarizer**:
        - Implements the BART model to generate concise summaries from longer texts.
        - Extracts key points to present essential information quickly and efficiently.
    ''')