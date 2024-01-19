#!/usr/bin/perl
#run scripts in terminal with perl [file path]

print "Hello, World!\n";

$foo = 10;

print "value of a is $a\n";
print 'value of a is $a\n';

$var = <<"EOF";
This is an example of multiline text that we can use with the << operator
it is very nice
EOF

print "\"$var\"\n";

$var = <<"EOF";
Datatypes have 3 starters,
scalars are $
arrays are @
hashes are %
EOF

print "$var\n";
print "\aalert\n";

print "now time for arrays\n";

@ages = (24,30,40);

print "$ages[0]\n";
print "$ages[1]\n";
print "@ages\n";

print "hashes\n"

%test = ('me',23,'you',42,'him',30);

$data{'me'} = 44;

print "%test\n";

delete %test 'me';




# stopped at arrays
#https://www.tutorialspoint.com/perl/perl_arrays.htm