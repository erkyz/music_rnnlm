addpath(genpath('../miditoolbox'));
path = 'music_data/ashover/test/';
files = dir(strcat(path, '*.mid'));
thres = 0.21; % empirically reached this.
max_ratio = 0.2; % one in five.
min_sections = 2;
fileID = fopen(strcat(path, 'segments.txt'),'w');

for file = files'
	a = readmidi(strcat(file.folder, '/', file.name));
	x = 1;
	prevnote = a(1);

	% Remove all the chord info for Nottingham.
	for note = a'
		if (x ~= 0 && note(1) < prevnote(1))
			a = a(1:x-1,:);
			break
		end
		prevnote = note;
		x = x+1;
	end

	segs = boundary(a);
	starts = find(segs > thres);
	if (length(starts) / length(a) < max_ratio && length(starts) >= min_sections)
		disp(file.name)
		fprintf(fileID, '%s\n', file.name);
		fprintf(fileID, '%d ', starts);
		fprintf(fileID, '\n');
	end

end

fclose(fileID)
