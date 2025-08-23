par :: String -> String
par a = "(" ++ a ++ ")"


data Poly = Var String | Poly :+: Poly | Poly :*: Poly | Lit Int | Poly :^: Int

showPoly :: Poly -> String
showPoly (Var a) = a
showPoly (a :+: b) = showPoly a ++ "+" ++ showPoly b
showPoly (a :*: b) = showPoly a ++ "*" ++ showPoly b
showPoly (Lit a) = show a
showPoly (x :^: n) = (par (showPoly x) ++ "^" ++ show n)

instance Show Poly where
    show e = showPoly e
