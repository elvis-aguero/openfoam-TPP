
import sys

def generate_geo_flat(H, D, mesh_size):
    R = D / 2.0
    
    geo_content = f"""
// Inputs (Flat Bottom)
H = {H};
R = {R};
lc = {mesh_size};

// Geometry using OpenCASCADE kernel for simple primitives
SetFactory("OpenCASCADE");

// Cylinder
Cylinder(1) = {{0, 0, 0, 0, 0, H, R}};

// Mesh Quality
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Physical Groups
eps = 1e-3;

// Top
Surface_Top[] = Surface In BoundingBox {{ -R-eps, -R-eps, H-eps, R+eps, R+eps, H+eps }};
Physical Surface("atmosphere") = Surface_Top[];

// Bottom
Surface_Bot[] = Surface In BoundingBox {{ -R-eps, -R-eps, -eps, R+eps, R+eps, eps }};

// Lateral (The rest)
All_Surfaces[] = Surface "*";
Lateral_Surfaces[] = {{}};

For i In {{0 : #All_Surfaces[]-1}}
    id = All_Surfaces[i];
    is_top = 0;
    is_bot = 0;
    
    For j In {{0 : #Surface_Top[]-1}}
        If (id == Surface_Top[j])
            is_top = 1;
        EndIf
    EndFor
    
    For k In {{0 : #Surface_Bot[]-1}}
        If (id == Surface_Bot[k])
            is_bot = 1;
        EndIf
    EndFor
    
    If (is_top == 0 && is_bot == 0)
        Lateral_Surfaces[] += {{id}};
    EndIf
EndFor

Physical Surface("walls") = {{Surface_Bot[], Lateral_Surfaces[]}};
Physical Volume("internalMesh") = {{1}};
"""
    return geo_content

def generate_geo_cap(H, D, mesh_size):
    R = D / 2.0
    
    geo_content = f"""
// Inputs (Spherical Cap)
H = {H};
R = {R};
lc = {mesh_size};

SetFactory("OpenCASCADE");

// Cylinder Body from z=0 to z=H
v1 = newv;
Cylinder(v1) = {{0, 0, 0, 0, 0, H, R}};

// Spherical Cap at z=0 (radius R)
// This sphere will fuse with the cylinder.
// The top half (z > 0) is inside the cylinder.
// The bottom half (z < 0) forms the hemispherical cap.
v2 = newv;
Sphere(v2) = {{0, 0, 0, R}};

// Fuse them
v_fused[] = BooleanUnion{{ Volume{{v1}}; Delete; }}{{ Volume{{v2}}; Delete; }};

// Mesh Quality
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

eps = 1e-3;

// Physical Surfaces

// Top Surface (at z=H)
Surface_Top[] = Surface In BoundingBox {{ -R-eps, -R-eps, H-eps, R+eps, R+eps, H+eps }};
Physical Surface("atmosphere") = Surface_Top[];

// Walls (Everything else)
All_Surfaces[] = Surface "*";
Wall_Surfaces[] = {{}};

For i In {{0 : #All_Surfaces[]-1}}
    id = All_Surfaces[i];
    is_top = 0;
    
    For j In {{0 : #Surface_Top[]-1}}
        If (id == Surface_Top[j])
            is_top = 1;
        EndIf
    EndFor
    
    If (is_top == 0)
        Wall_Surfaces[] += {{id}};
    EndIf
EndFor

Physical Surface("walls") = Wall_Surfaces[];
Physical Volume("internalMesh") = v_fused[];
"""
    return geo_content

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 generate_mesh.py <Height> <Diameter> <MeshSize> [GeometryType: flat/cap]")
        sys.exit(1)
        
    H = float(sys.argv[1])
    D = float(sys.argv[2])
    lc = float(sys.argv[3])
    
    # Default to flat
    geo_type = "flat"
    if len(sys.argv) >= 5:
        geo_type = sys.argv[4]
    
    if geo_type == "cap":
        geo_content = generate_geo_cap(H, D, lc)
        print(f"Generating Spherical Cap Geometry (D={D}, H={H})")
    else:
        geo_content = generate_geo_flat(H, D, lc)
        print(f"Generating Flat Bottom Geometry (D={D}, H={H})")
    
    with open("cylinder.geo", "w") as f:
        f.write(geo_content)
    
    print("cylinder.geo created.")
