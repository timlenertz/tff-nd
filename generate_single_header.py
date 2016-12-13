#!/usr/bin/python

import sys
import os
import re
import sets

included_files = sets.Set()


copyright_prefix = open('copyright.txt', 'r').read()

prefix = copyright_prefix + """
//--- TFF nd single-header version
//--- generated using generate_single_header.py

#define TFF_ND_STANDALONE

"""

def include_file(filename):	
	filename = os.path.abspath(filename)
	if filename in included_files: 
		sys.stdout.write("//--- already included")
		return
		
	included_files.add(filename)
	
	dirname = os.path.dirname(filename)
	
	for line in open(filename, 'r'):
		local_include_obj = re.search(r'\s*#include\s+"([A-z0-9_./]+)"', line, re.I)
		if local_include_obj:
			inc_rel_filename = local_include_obj.group(1)
			inc_filename = os.path.join(dirname, inc_rel_filename)
			if(os.path.isfile(inc_filename)):
				sys.stdout.write("\n//--- include " + inc_rel_filename + "\n")
				include_file(inc_filename)
				sys.stdout.write("\n//--- end include " + inc_rel_filename + "\n")
			else:
				sys.stdout.write(line)
		else:
			sys.stdout.write(line)
		

sys.stdout.write(prefix)
include_file(os.path.abspath('src/nd.h'))

