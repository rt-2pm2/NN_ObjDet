#!/bin/zsh
#
# This script processed the downloaded files making the name more uniform
#
HOME=`pwd`
IN='raw'
OUT='preprocessed'
CAT_LIST=()
COUNTER=0

printf "////===== Preprocessing the image names =====//// \n\n" 

# Load the list of categories from the ./raw folder
if [ $# -eq 1 ]; then
	TARGET_CAT=$1
	printf "Selected category: %s" $TARGET_CAT
	PATH2CAT=$IN/$TARGET_CAT/

	printf "=================================\n" 
	printf "Added category: %s \n" ${TARGET_CAT}
	OUTPATH=$OUT/$TARGET_CAT
	mkdir -p $OUTPATH
	# Rename files
	printf "\t-- Pre-processing files of category %s ...\n" $TARGET_CAT
	for ofilename in ${PATH2CAT}*jpg; do
		COUNTER=$((COUNTER+1))
		printf "\tProcessing file No [%d]: %s \n" \
			$COUNTER ${ofilename#$PATH2CAT}
	
		# Removing trailing part and renaming
		#nfilename=$(echo "$ofilename" | cut -d'.' -f1 | tr -d ' ').jpg
	
		# Creating a new filename
		nfilename=${TARGET_CAT}${COUNTER}.jpg
		printf "\tRenaming file %s --> %s \n\n" \
			${ofilename#$PATH2CAT} $nfilename

		cp -i $ofilename $OUTPATH/$nfilename
	done
else
	# Look for folders in ./$IN/
	printf "Looking for categories in the \"%s\" folder \n" $IN
	for PATH2CAT in $IN/*/; do
		# Strip out the path
		TARGET_CAT=${PATH2CAT#$IN/}
		# Remove the trailing "/"
		TARGET_CAT=${TARGET_CAT%/}
		CAT_LIST+=(${TARGET_CAT%/})

		printf "=================================\n" 
		printf "Added category: %s \n" ${TARGET_CAT}
		OUTPATH=$OUT/$TARGET_CAT
		mkdir -p $OUTPATH
		# Rename files
		printf "\t-- Pre-processing files of category %s ...\n" $TARGET_CAT
		for ofilename in ${PATH2CAT}*jpg; do
			COUNTER=$((COUNTER+1))
			printf "\tProcessing file No [%d]: %s \n" \
				$COUNTER ${ofilename#$PATH2CAT}
		
			# Removing trailing part and renaming
			#nfilename=$(echo "$ofilename" | cut -d'.' -f1 | tr -d ' ').jpg
		
			# Creating a new filename
			nfilename=${TARGET_CAT}${COUNTER}.jpg
			printf "\tRenaming file %s --> %s \n\n" \
				${ofilename#$PATH2CAT} $nfilename

			cp -i $ofilename $OUTPATH/$nfilename
		done
	done
fi
