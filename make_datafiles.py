import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

class FileTokenizer(object):
  """Object for making 'bin' files"""

  def __init__(self, stories_dir, out_dir):
    self.__stories_dir = stories_dir
    self.__out_dir = out_dir
    self.__tokenized_stories_dir = os.path.join(self.__out_dir, "tokenized_stories_dir")
    self.__finished_files_dir = os.path.join(self.__out_dir, "finished_files") 
    self.__chunks_dir = os.path.join(self.__finished_files_dir, "chunked")

  def get_stories_dir(self):
    return self.__stories_dir

  def chunk_file(self, set_name):
    # in_file = 'finished_files/%s.bin' % set_name 
    in_file = os.path.join(self.__finished_files_dir, '%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
      chunk_fname = os.path.join(self.__chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
      with open(chunk_fname, 'wb') as writer:
        for _ in range(CHUNK_SIZE):
          len_bytes = reader.read(8)
          if not len_bytes:
            finished = True
            break
          str_len = struct.unpack('q', len_bytes)[0]
          example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, example_str))
        chunk += 1


  def chunk_all(self):
    # Make a dir to hold the chunks
    if not os.path.isdir(self.__chunks_dir):
      os.mkdir(self.__chunks_dir)
    # Chunk the data
    # for set_name in ['train', 'val', 'test']:
    for set_name in ['test']:
      print("Splitting %s data into chunks..." % set_name)
      self.chunk_file(set_name)
    print("Saved chunked data in %s" % self.__chunks_dir)


  def tokenize_stories(self):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (self.__stories_dir, self.__tokenized_stories_dir))
    stories = os.listdir(self.__stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
      for s in stories:
        f.write("%s \t %s\n" % (os.path.join(self.__stories_dir, s), os.path.join(self.__tokenized_stories_dir, s)))
    command = ['java','-cp','stanford-corenlp-3.9.1.jar','edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), self.__stories_dir, self.__tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(self.__stories_dir))
    num_tokenized = len(os.listdir(self.__tokenized_stories_dir))
    if num_orig != num_tokenized:
      raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (self.__stories_dir, self.__tokenized_stories_dir))


  def read_text_file(self, text_file):
    lines = []
    with open(text_file, "r") as f:
      for line in f:
        lines.append(line.strip())
    return lines


  def fix_missing_period(self, line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


  def get_art_abs(self, story_file):
    lines = self.read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [self.fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
      if line == "":
        continue # empty line
      elif line.startswith("@highlight"):
        next_is_highlight = True
      elif next_is_highlight:
        highlights.append(line)
      else:
        article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


  def write_to_bin(self, out_file):
    # story_fnames = [name for name in os.listdir(tokenized_stories_dir) if os.path.isfile(tokenized_stories_dir+'\\'+name) ]
    story_fnames = [name for name in os.listdir(self.__tokenized_stories_dir)]
    num_stories = len(story_fnames)
    # print(story_fnames)
    # for idx,s in enumerate(story_fnames):
    #   print(idx,s)


    with open(out_file, 'wb') as writer:
      for idx,s in enumerate(story_fnames):
        if idx % 1000 == 0:
          print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(self.__tokenized_stories_dir, s)):
          story_file = os.path.join(self.__tokenized_stories_dir, s)
          print(story_file)

        else:
          print('Error: no data.')

    #     # Get the strings to write to .bin file
        article, abstract = self.get_art_abs(story_file)
        # print('article:', article)
        # print('abstract:', abstract)

        # Write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

    print("Finished writing file %s\n" % out_file)
    print("Deleting files from %s" % self.__tokenized_stories_dir)
    for name in story_fnames:
    	if os.path.isfile(os.path.join(self.__tokenized_stories_dir, name)):
    		os.remove(os.path.join(self.__tokenized_stories_dir, name))


  def make_binfiles(self):
    # if len(sys.argv) != 3:
    #   print("USAGE: python make_datafiles.py <stories_dir> <out_dir>")
    #   sys.exit()
    # stories_dir = sys.argv[1]
    # out_dir = sys.argv[2]
    # print(stories_dir)
    # tokenized_stories_dir = "data/tokenized_stories_dir"
    # finished_files_dir = "data/finished_files"
    
    
  	
    # Create some new directories
    if not os.path.exists(self.__tokenized_stories_dir): os.makedirs(self.__tokenized_stories_dir)
    if not os.path.exists(self.__finished_files_dir): os.makedirs(self.__finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    self.tokenize_stories()

    # Read the tokenized stories, do a little postprocessing then write to bin files
    self.write_to_bin(os.path.join(self.__finished_files_dir, "test.bin"))

    # # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    self.chunk_all()
  
  