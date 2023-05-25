#!/usr/bin/perl
open(TEST,"<$ARGV[0]")||die("CANT OPNE TEST\n");
open(TRUE,"<$ARGV[1]")||die("CANT OPNE TRUE\n");

@text_true = <TRUE>;
@text_test = <TEST>;

$n = 0;
$hit = 0;
foreach $line_test (@text_test){
    chomp($line_test);
    my @sbuf_test = split / /, $line_test;

    $line_true = $text_true[$n];
    chomp($line_true);
    my @sbuf_true = split / /, $line_true;

    if ($sbuf_test[1] eq $sbuf_true[1]) {
	$hit = $hit + 1;
    }
    
    #print "$line_result\n";
    $n = $n + 1;
}

$total = $n;
$acc = $hit / $total * 100;
print "============ Results Analysis ===========\n";
print "Test: $ARGV[0]\n";
print "True: $ARGV[1]\n";

print "Accuracy: ", sprintf("%.2f", ${acc}), "\%\n";
print "Hit: $hit, Total: $total\n";
print "=========================================\n";

close(TRUE);
close(TEST);

