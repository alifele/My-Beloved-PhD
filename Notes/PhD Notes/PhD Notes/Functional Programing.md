
Important resources
* https://www.cs.ox.ac.uk/jeremy.gibbons/publications/origami.pdf
* https://people.cs.nott.ac.uk/pszgmh/fold.pdf



Building a unified picture
* Initial algebra: 
	* Various finite data structures used in programming, such as lists and trees, can be obtained as initial algebras of specific endofunctors. While there may be several initial algebras for a given endofunctor, they are unique up to isomorphism, which informally means that the "observable" properties of a data structure can be adequately captured by defining it as an initial algebra.
	* Initial algebra is one of the main things that I need to study carefully to understand what is happening. Because a lot of the algebraic data types like lists, trees, graphs, etc fall in this category. For example a expression tree like $(1+2)*3$ or $2*(1+3)+(3+1)*3$ are algebraic types of the form $$ F(X) = \mathbb{Z} + X\times X + X\times X. $$
	* Also a recursive expression like 
	  ```Haskel
	  data Exp = Lit n | Add Exp Exp | Mul Exp Exp 
	  ```
	  is a more compact way of capturing this.
	* #OpenQuestion One of the open questions that I have is that if representing a tree as a pretty-print, an actual graph, or as a recursive definition are the same, then what structure preserving map exists between them?