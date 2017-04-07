% minFunc
fprintf('Compiling minFunc files...\n');
mex -o minFunc/compiled/mcholC.mex minFunc/mex/mcholC.c
mex -o minFunc/compiled/lbfgsC.mex minFunc/mex/lbfgsC.c
mex -o minFunc/compiled/lbfgsAddC.mex minFunc/mex/lbfgsAddC.c
mex -o minFunc/compiled/lbfgsProdC.mex minFunc/mex/lbfgsProdC.c

