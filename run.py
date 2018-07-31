from make_datafiles import FileTokenizer
import run_summarization as run

# Makes bin files
datafiles = FileTokenizer('stories', 'output')
datafiles.make_binfiles()

# Runs summarizer
run.runfiles('output/finished_files/test.bin','vocab','../','decode', datafiles)