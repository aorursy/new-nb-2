# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imgNames = ["aac893a91","51f1be19e","f5a1f0358","796707dd7","51b3e36ab","348a992bb","cc3532ff6","2fd875eaa","cb8d261a3","53f253011"]

result = ["0.7335314 370 491 440 644 0.7324162 622 896 694 1024 0.73239845 192 0 497 136 0.7289894 646 953 668 988 0.72781235 463 823 534 978 0.72715443 339 424 481 717 0.72614396 583 818 731 1024 0.72564435 488 883 511 916 0.7228437 335 13 358 46 0.7218616 347 793 648 1004 0.7216297 310 0 382 108 0.7205864 636 996 660 1018 0.7198859 645 943 701 973 0.7184655 229 827 380 943 0.71833134 634 717 924 930 0.71716934 318 0 357 61 0.71360236 423 556 466 578 0.71273345 69 555 362 767 0.7120335 635 931 673 1007 0.7120062 398 609 416 626 0.71181566 650 786 696 811 0.71156365 766 803 789 837 0.7113237 741 743 812 898 0.7109132 152 775 444 987 0.71052843 494 866 536 887 0.70994496 635 839 654 857 0.70758355 477 919 495 935 0.70403063 332 0 382 26 0.7032378 321 51 342 71 0.7022946 477 852 515 929 0.7016563 394 548 418 581 0.70157135 113 57 136 90",

          "0.7380025 224 118 373 230 0.7337337 781 751 931 866 0.7245482 265 468 413 583 0.723774 704 885 776 1024 0.7196501 419 427 723 629 0.7195352 760 254 1024 463 0.71920455 583 853 893 1024 0.71880287 697 713 988 927 0.71762925 801 71 947 183 0.7172307 108 882 420 1024 0.71650374 724 28 1024 230 0.71639127 71 13 94 47 0.7162473 187 419 490 623 0.7150792 0 0 242 139 0.7150042 48 0 116 104 0.71393883 499 471 650 586 0.7138737 728 942 751 975 0.71321666 0 326 182 536 0.71154827 714 998 736 1019 0.71113753 942 681 965 715 0.71048135 652 922 723 1024 0.7097791 0 348 61 500 0.7092569 16 406 39 439 0.7091467 568 469 608 489 0.70868987 861 113 884 146 0.707555 156 72 447 283 0.7073515 334 471 375 492 0.70690006 554 520 570 536 0.70552135 318 522 335 539 0.7050257 240 991 260 1012 0.7049147 247 940 296 966 0.70370436 256 973 280 1006 0.702729 609 640 632 673 0.7019049 884 312 926 334 0.7014557 799 614 1024 826 0.70012814 870 365 887 382",

          "0.73219174 493 774 563 926 0.7303555 620 696 692 849 0.7272347 293 450 443 562 0.72557503 439 335 735 541 0.72473365 411 132 481 285 0.7237856 0 790 190 1003 0.7229646 136 259 434 466 0.72225827 703 136 726 169 0.72164214 0 414 282 622 0.7214427 219 399 515 612 0.72101295 594 141 894 354 0.7208887 45 177 344 384 0.7204982 465 700 610 995 0.7198614 213 305 360 418 0.7189209 496 670 804 868 0.71866494 298 100 596 307 0.7176518 425 0 734 158 0.71715856 20 878 43 912 0.71628016 115 512 415 713 0.7159142 821 383 892 536 0.715585 435 191 458 225 0.7151951 731 229 754 263 0.71511793 835 431 1024 632 0.7150175 468 302 539 451 0.7141914 14 863 64 889 0.71383756 5 914 25 935 0.7137699 569 390 611 411 0.713736 231 540 302 691 0.7135137 174 233 216 254 0.7118676 645 757 668 791 0.71172965 161 285 178 302 0.71156603 714 354 1005 555 0.71115255 274 345 297 379 0.71098405 250 311 292 332 0.71097434 956 458 1024 607 0.71088743 555 443 572 460 0.7103383 491 311 535 334 0.70923847 240 362 256 379 0.7089199 477 364 495 382 0.70880514 354 488 376 520 0.7082034 143 744 292 858 0.70757324 647 704 689 726 0.70733833 96 470 137 491 0.7073219 405 154 446 174 0.70720536 737 241 778 261 0.7058434 330 470 372 491 0.7058409 84 521 101 538 0.70577216 714 294 731 310 0.70538616 634 756 651 773 0.7052899 397 204 412 220 0.70455796 823 605 846 638 0.7044983 319 522 336 538 0.703817 676 526 976 718 0.7025502 478 782 519 802 0.70150316 799 547 867 698 0.7007098 542 0 614 135 0.7001997 474 830 490 846",

          "0.73094565 366 0 435 102 0.7277978 918 301 987 454 0.7246667 391 8 414 42 0.7207291 616 767 913 977 0.7207287 241 0 558 133 0.71910566 941 361 964 396 0.7180159 0 382 191 597 0.7170588 693 816 842 929 0.7168389 0 412 70 563 0.7161538 885 229 1024 522 0.70977783 378 0 416 61 0.70928985 62 78 212 192 0.7087947 0 33 285 236 0.7068233 142 322 436 530 0.706622 761 855 784 889 0.7059267 21 473 68 498 0.70539486 5 526 25 545 0.70339715 415 0 468 29 0.7010248 545 819 568 852 0.70081955 401 52 424 73 0.7005451 739 870 779 890",

          "0.73711365 446 316 593 429 0.73609495 485 539 554 692 0.7331419 923 208 945 242 0.73183674 404 388 474 542 0.73103666 271 697 293 732 0.7294366 429 447 452 481 0.72943276 605 756 754 869 0.72898287 0 856 204 1024 0.7282505 451 960 474 993 0.72786826 453 465 597 758 0.7267131 8 890 76 1024 0.7256327 356 0 657 202 0.72556716 341 459 413 612 0.7245317 668 797 691 831 0.7244249 247 643 315 793 0.7242099 25 781 98 936 0.72335476 5 994 33 1020 0.72268784 0 556 203 769 0.72259617 861 168 1012 281 0.7223871 509 599 531 634 0.72221446 305 382 452 684 0.7220033 309 870 612 1024 0.7215398 14 579 85 731 0.7215308 11 942 73 976 0.72133845 369 267 675 468 0.7212455 427 900 499 1024 0.7202341 910 182 948 259 0.72022176 541 715 835 927 0.7201564 0 765 221 954 0.7188859 132 617 432 827 0.7185384 0 0 201 204 0.7181881 791 130 1024 324 0.71742636 110 833 260 947 0.7173097 471 19 544 169 0.716726 266 91 568 300 0.7163868 437 930 475 1007 0.715753 533 208 556 241 0.7156034 412 156 458 180 0.7153679 418 418 456 496 0.7140717 532 0 832 203 0.71402955 174 876 196 909 0.71385396 399 208 418 227 0.7134319 372 311 516 607 0.7126429 830 546 1024 761 0.7121843 501 81 545 105 0.7116597 480 135 499 153 0.7106249 4 678 24 699 0.71022993 13 627 62 653 0.7092092 13 17 82 167 0.70812666 508 149 581 302 0.7071413 399 939 450 966 0.7070324 37 641 60 675 0.70684355 397 990 417 1010 0.7067762 260 674 299 751 0.7067059 647 785 688 806 0.706471 486 311 531 335 0.7060792 477 361 495 380 0.7048484 485 546 527 567 0.70461667 2 778 50 802 0.7044301 242 765 259 781 0.7043964 634 838 651 854 0.70368737 494 75 517 109 0.7033957 476 595 493 612 0.70263106 3 822 22 842 0.70071405 497 577 535 652",

          "0.7359927 127 887 195 1024 0.7345171 455 614 526 767 0.734465 0 306 129 419 0.73411083 108 173 181 326 0.73240596 285 298 354 452 0.73010534 420 540 566 839 0.7299192 732 208 878 321 0.7296752 887 553 1024 664 0.7292843 491 934 640 1024 0.72911906 583 435 730 550 0.72790974 0 870 192 1024 0.7276939 19 960 42 993 0.72726077 861 807 934 959 0.7271344 385 177 457 328 0.72674286 0 257 212 464 0.72614616 689 347 760 501 0.725601 49 228 119 380 0.7255255 374 812 445 963 0.72540903 647 159 956 369 0.72487646 408 885 724 1024 0.7242895 275 804 573 1014 0.72358197 745 777 1024 992 0.72272515 613 408 907 617 0.72223985 0 418 218 622 0.72117454 70 108 214 403 0.7210626 5 992 30 1016 0.7210296 269 148 575 360 0.7208185 74 288 97 322 0.72054887 801 510 1024 710 0.71996266 9 939 67 971 0.71928173 253 227 395 521 0.7182302 137 234 159 267 0.7181019 433 918 456 952 0.71789217 418 236 464 260 0.7177379 311 359 333 395 0.7172904 709 409 731 442 0.7169966 8 307 61 336 0.7169447 657 471 703 496 0.71642715 400 291 419 310 0.71625704 809 236 855 260 0.715721 965 805 987 839 0.71569186 4 358 26 380 0.7156864 895 933 967 1024 0.71530074 955 544 1011 576 0.71528435 95 817 235 1024 0.71505624 952 595 975 618 0.7147518 731 950 881 1024 0.7145405 407 861 477 1015 0.7144481 892 868 938 893 0.71388716 636 527 656 546 0.713305 557 993 580 1015 0.71302164 792 289 812 307 0.71297425 873 922 892 941 0.71296185 94 233 138 255 0.71287864 776 911 1024 1024 0.71277744 566 940 619 969 0.71258867 416 865 459 887 0.71246445 82 285 99 302 0.7123456 0 901 67 1024 0.7117621 956 782 1014 815 0.71159047 313 142 386 293 0.7114766 328 152 372 175 0.71090865 952 834 977 859 0.7106793 940 745 1011 902 0.71061575 170 616 193 651 0.7103053 399 918 417 936 0.70962846 319 203 337 221 0.7094268 716 468 757 487 0.7094166 180 946 234 976 0.7093512 570 64 593 99 0.7090741 964 642 1024 791 0.7089386 574 0 875 203 0.7089019 362 478 661 689 0.70875216 791 245 815 279 0.7083992 655 269 800 565 0.7075011 552 974 576 1008 0.70741445 710 518 724 534 0.7072516 590 778 613 812 0.707232 63 262 101 337 0.70712584 163 1000 187 1021 0.7066732 491 624 535 647 0.70653856 333 312 377 336 0.706046 847 610 1024 822 0.7057514 478 677 497 695 0.7056396 146 558 218 708 0.7054571 691 12 762 162 0.7052934 644 470 667 503 0.7043316 4 931 42 1007 0.70413727 988 697 1010 731 0.70379066 8 465 57 491 0.7032758 3 516 23 536 0.70258725 483 675 506 711 0.70161694 37 0 350 168 0.7010424 27 529 331 737 0.7010149 414 469 486 624 0.70000654 410 234 433 267",

          "0.7348467 398 4 436 82 0.73387384 0 638 139 749 0.7337318 410 33 432 66 0.7322321 644 460 666 495 0.7320953 925 100 996 253 0.72817665 385 0 457 127 0.727834 805 683 876 835 0.7266642 892 30 1024 323 0.7265347 748 500 771 534 0.72556484 264 0 575 152 0.7255475 700 286 849 400 0.7247926 503 566 575 716 0.7246662 531 621 554 655 0.7245085 580 422 726 535 0.72271127 729 704 800 858 0.721524 765 323 787 358 0.7205721 836 0 1024 147 0.7201406 0 415 196 617 0.71985155 534 823 684 936 0.71927303 627 238 922 452 0.71892875 612 408 911 618 0.71809417 676 828 975 1024 0.718022 827 741 851 774 0.7171323 688 655 992 860 0.7167869 397 538 688 749 0.71678156 387 366 683 560 0.7165632 0 591 224 790 0.71454936 103 600 175 748 0.71413785 467 777 758 989 0.7139724 404 0 456 22 0.7135842 718 468 762 490 0.712742 399 45 420 66 0.71258146 712 519 728 536 0.7125153 964 157 1018 188 0.71106875 618 460 689 514 0.7110309 0 397 147 513 0.7107206 795 705 838 727 0.710282 952 210 974 232 0.7101339 311 253 382 403 0.70844114 516 595 554 672 0.70843303 789 754 805 771 0.70811176 687 460 833 572 0.7078451 721 311 765 334 0.7070317 675 0 974 189 0.7065054 714 363 731 381 0.70506394 2 622 51 648 0.70459735 71 832 221 944 0.7045444 983 27 1005 61 0.70446503 3 668 22 688 0.7040175 54 676 77 710 0.70351106 637 534 656 551 0.7029268 198 226 496 435 0.7026249 665 480 709 504 0.70186025 493 629 532 649 0.7010847 575 868 615 888 0.7008955 477 684 493 700 0.7001043 960 0 1024 117",

          "0.73838776 446 23 516 176 0.73752123 103 564 249 677 0.7373318 413 0 555 244 0.7341207 802 685 873 839 0.7328635 745 852 817 1002 0.73284084 93 10 241 122 0.7313338 772 620 911 913 0.73048526 40 934 64 968 0.7292695 417 935 565 1024 0.72890127 490 540 512 574 0.7287567 14 876 85 1024 0.728132 94 818 239 930 0.7277314 34 764 57 798 0.7271924 153 51 175 85 0.7271744 0 0 134 88 0.72685754 122 0 429 136 0.7258135 929 72 953 105 0.72524756 8 705 77 859 0.7242849 154 858 176 892 0.7242761 24 909 62 987 0.72370964 0 845 213 1024 0.72331095 0 0 215 138 0.72314453 436 814 740 1018 0.72258836 392 759 464 909 0.72250795 10 771 317 981 0.72121215 338 889 652 1024 0.7207884 827 742 850 775 0.7202072 125 851 199 908 0.71956223 11 523 326 715 0.7179655 913 851 980 1002 0.71782 635 833 929 1024 0.7177749 124 44 198 101 0.71764547 763 174 786 207 0.71748394 796 0 1024 193 0.7163245 866 32 1014 147 0.71585065 565 863 609 885 0.7153203 469 83 492 119 0.715268 461 537 534 593 0.71449775 555 914 572 931 0.7139204 274 729 576 936 0.71380913 826 591 1024 788 0.7133993 418 816 442 850 0.7133041 936 910 959 946 0.713241 0 682 200 875 0.7114759 3 0 61 25 0.7114492 932 739 1004 891 0.7114067 351 456 649 666 0.7109083 5 43 28 68 0.71036506 174 546 217 569 0.71023065 951 620 1006 650 0.7100838 499 80 542 101 0.7093424 42 15 66 49 0.7093225 5 999 29 1021 0.70842564 196 876 239 898 0.70837086 478 134 495 151 0.7083526 950 671 972 694 0.7082288 411 786 451 806 0.7081498 161 597 179 615 0.7073722 629 88 920 297 0.70729965 164 928 183 945 0.7057014 483 938 535 967 0.7056474 397 837 413 853 0.7052144 477 990 499 1011 0.7052076 488 979 511 1013 0.7043848 771 911 796 945 0.7040163 253 995 276 1024 0.70317686 813 707 1024 921 0.7031707 958 794 981 829 0.70314825 380 295 678 483 0.70282954 801 706 840 725 0.70179707 910 42 947 121 0.70166105 24 734 62 810 0.70076156 791 756 805 772",

          "0.7358016 825 253 847 287 0.7353108 252 753 399 866 0.7350454 843 142 914 295 0.73162854 811 220 849 299 0.7310648 604 227 674 380 0.72809577 771 111 920 408 0.7276649 572 151 718 447 0.7264935 467 775 539 926 0.7259892 798 193 868 347 0.7258986 27 858 98 1010 0.7255168 697 138 768 288 0.72542846 787 737 810 771 0.7250652 684 697 707 732 0.7243919 417 98 567 207 0.7223749 170 705 478 917 0.7221982 332 60 642 246 0.72176546 629 287 652 321 0.72143555 0 826 217 1024 0.72083735 347 748 655 946 0.71971434 768 925 839 1024 0.7191323 662 65 807 366 0.71870816 761 679 831 830 0.7176539 803 232 850 256 0.71755534 548 608 843 814 0.71561944 341 790 385 813 0.7154281 792 286 810 304 0.7152372 491 834 514 868 0.7145621 355 405 650 615 0.71307766 320 843 339 860 0.7129668 424 453 576 561 0.71172804 495 473 536 494 0.7106414 486 783 528 804 0.71045 658 637 730 788 0.7103981 477 525 494 542 0.7099096 649 898 963 1024 0.7094853 476 833 493 850 0.7083945 0 759 59 909 0.7080209 772 714 810 790 0.7070945 58 843 366 1024 0.70546454 885 153 928 176 0.70519286 646 230 690 253 0.7048931 731 617 870 913 0.70425254 871 204 890 222 0.70393944 634 282 652 300 0.7034788 615 261 653 338 0.7026406 725 153 767 175 0.7023374 5 858 55 883 0.7022702 713 204 731 222 0.7010886 3 906 23 927 0.7009368 674 676 712 750",

          "0.7341572 645 99 715 252 0.73243964 253 806 324 960 0.7316958 377 819 524 933 0.73126125 810 608 882 762 0.73106056 614 28 756 321 0.729215 221 738 362 1024 0.7276746 775 540 918 836 0.7272455 144 313 292 429 0.72492117 173 662 196 695 0.7241424 40 867 349 1024 0.7231547 350 130 659 323 0.7220289 13 30 160 141 0.72115713 299 771 597 982 0.72102636 195 20 503 226 0.7200982 510 320 811 528 0.720023 39 581 338 784 0.71860445 71 641 93 676 0.7180582 169 941 224 970 0.71777886 70 263 366 468 0.7177464 162 994 184 1016 0.7175767 289 264 586 466 0.71637106 148 605 221 756 0.7151352 260 380 560 590 0.7147818 258 867 301 890 0.7147239 525 352 596 504 0.71382254 507 649 807 857 0.7137921 617 346 689 498 0.71357596 241 921 259 938 0.7130016 795 625 839 646 0.71210736 7 602 156 714 0.71202344 835 800 1024 1011 0.71186054 650 157 695 180 0.71118397 331 77 376 100 0.7106871 791 673 807 691 0.7105241 648 739 671 773 0.7102279 319 128 338 146 0.7100461 635 211 653 228 0.70984316 165 73 209 95 0.70917594 162 636 200 711 0.7086403 830 665 853 698 0.70858806 0 338 209 544 0.70820504 160 122 176 140 0.7075687 473 546 763 754 0.707442 649 391 693 413 0.70627606 0 0 242 193 0.7060838 75 67 97 101 0.70585096 634 445 652 462 0.7057677 826 157 1024 368 0.70552427 175 626 217 648 0.7053997 419 868 461 889 0.70498234 39 426 63 459 0.7045057 937 767 960 801 0.70448387 0 728 134 838 0.7033882 495 211 518 245 0.70331335 161 678 178 695 0.70310974 399 923 416 939 0.70256 211 349 234 382 0.7020815 255 550 295 570 0.70173025 486 151 527 173 0.701595 169 311 210 331 0.7007187 476 201 493 218 0.7006153 17 372 87 524 0.7005058 914 708 983 861 0.7004855 579 394 623 417 0.70028186 161 361 177 377"]

outputCsv = pd.DataFrame(columns=["image_id", "PredictionString"])

for i in range(len(imgNames)):

    newPd = pd.DataFrame({"image_id":imgNames[i], "PredictionString":result[i]},index=[0])

    outputCsv = pd.concat((outputCsv, newPd), ignore_index=True, sort=False)

outputCsv.to_csv("/kaggle/working/submission.csv",index=False)