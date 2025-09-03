using Pkg

# Returns a list of all of the installed packages.
function getAvlblPkgs()
	return [dep.name for (uuid, dep) in Pkg.dependencies() if dep.is_direct_dep]
end

# Checks the needed packages one by one and installs if not already installed.
function ensure_packages(needed_packages::String...)
	printstyled("Handling the dependencies...\n"; color = :magenta)
	# all of the available packages
	allpkgs = getAvlblPkgs()
	for pkg in needed_packages
		if pkg in allpkgs
			printstyled(pkg; bold = true)
			printstyled(" is already installed.\n"; color = :blue)
		else
			printstyled("Installing "; color = :green)
			printstyled(pkg; bold = true)
			printstyled("...\n"; color = :green)
			Pkg.add(pkg)
		end
	end
end
