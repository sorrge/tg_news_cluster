//    Copyright 2020 sorrge
//
//    This file is part of tg_news_cluster.
//
//    tg_news_cluster is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    tg_news_cluster is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with tg_news_cluster.  If not, see <https://www.gnu.org/licenses/>.
#include <sys/prctl.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>

using namespace std;


void die_when_parent_process_dies() {
	prctl(PR_SET_PDEATHSIG, SIGINT);
    if (getppid() == 1)
        raise(SIGINT);

    //cout << getpid() << " death on parent [" << getppid() << "] death activated" << endl;
}
