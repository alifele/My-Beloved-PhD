% labdiary.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{labdiary}[2023/11/06 Lab Diary custom class]

% Load the base class

\LoadClass[11pt,openany]{book}  % 'openany' option added


% Load required packages
\RequirePackage{geometry}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{microtype}
\RequirePackage{graphicx}
\RequirePackage{lipsum}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{color}
\RequirePackage{xcolor}
%\RequirePackage{12}
%\RequirePackage{palatino}
\RequirePackage{fontspec}



% Define colors
\definecolor{SectionColor}{rgb}{0.1,0.3,0.5}
\definecolor{SubSectionColor}{rgb}{0.2,0.4,0.6}

% Set page margins
\geometry{left=1in, right=1in, top=1in, bottom=1in}

% Customize chapter title
\titleformat{\chapter}[display]
{\normalfont\huge\bfseries\color{SectionColor}}
{\chaptertitlename\ \thechapter}{20pt}{\Huge}

% Customize section titles
\titleformat{\section}
{\normalfont\Large\bfseries\color{SectionColor}}
{\thesection}{1em}{}

% Customize subsection titles
\titleformat{\subsection}
{\normalfont\large\bfseries\color{SubSectionColor}}
{\thesubsection}{1em}{}

% Customize header and footer
\fancyhf{}
\fancyhead[LE,RO]{\bfseries\thepage}
\fancyhead[RE]{\bfseries\leftmark}
\fancyhead[LO]{\bfseries\rightmark}
\pagestyle{fancy}

%% Define the title page
%\renewcommand{\maketitle}{
%	\begin{titlepage}
%		\centering
%		{\Huge\bfseries\color{SectionColor} \@title \par}
%		\vspace{2cm}
%		{\Large\itshape \@author \par}
%		\vspace{1cm}
%		{\Large \@date \par}
%	\end{titlepage}
%}
%
\renewcommand{\maketitle}{
	\begin{titlepage}
		\centering
		{\Huge\bfseries\color{SectionColor} \@title \par}
		\vspace{2cm}
		{\Large\itshape \@author \par}
		\vspace{1cm}
		{\Large \@date \par}
		\thispagestyle{empty} % Suppresses page numbering on the title page
		\pagenumbering{gobble} % Suppresses page numbers
	\end{titlepage}
	\pagenumbering{arabic} % Resumes normal page numbering
}


\definecolor{customframecolor}{RGB}{0, 126, 140}
% Define a new command for custom section formatting
\newcommand{\newreviewsection}[3]{ % Three arguments: title, author, date
	\vspace{13pt}
	\refstepcounter{section} % Increment the section counter
	\begin{tcolorbox}[colback=white, colframe=customframecolor, title=\textbf{Paper Summary}]
		\textbf{Title:} #1 \\
		\textbf{Author(s):} #2 \\
		\textbf{Published:} #3
	\end{tcolorbox}
}






% End of class file
