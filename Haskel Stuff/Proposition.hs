data Prop = Var String | F | T | Not Prop | Prop :&: Prop | Prop :|: Prop deriving Eq

par :: String -> String
par a = "(" ++ a ++ ")"


showProp :: Prop -> String
showProp (Var a) = a
showProp F = "F"
showProp T = "T"
showProp (Not a) = par ("!" ++ (showProp a))
showProp (a :&: b) = par (showProp a ++ "&" ++ showProp b)
showProp (a :|: b) = par (showProp a ++ "|" ++ showProp b)



type Valn = String -> Bool

evalProp :: Prop -> Valn -> Bool
evalProp (Var a) val = val a
evalProp F _ = False
evalProp T _ = True
evalProp (Not a) v = not (evalProp a v)
evalProp (a :&: b) v = (evalProp a v) && (evalProp b v)
evalProp (a :|: b) v= (evalProp a v) || (evalProp b v)



myVals :: Valn
myVals "a" = False






instance Show Prop where
    show e = showProp e

