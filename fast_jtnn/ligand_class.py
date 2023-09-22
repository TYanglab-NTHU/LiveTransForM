class ligands_pair(object):
    def __init__(self):
        self.axial = []
        self.equ = []
        self.filepath = ''
        self.refcode = ''
        
    def recover_axil(self,axils):
        for axil in axils:
            self.axial.append(axil)
    def recover_equ(self,equs):
        for equ in equs:
            self.equ.append(equ)