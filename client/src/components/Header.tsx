import { SignInButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";

const Header = () => {
  return (
    <div className="pb-4 flex justify-between">
      <h1 className="text-2xl font-bold text-primary">IKMS</h1>
      <div className="flex items-center">
        <SignedIn>
          <UserButton
          />
        </SignedIn>
        <SignedOut>
          <SignInButton
            mode="modal"
          >
            <Button variant="default" className="font-semibold pointer-events-auto">
              Sign in
            </Button>
          </SignInButton>
        </SignedOut>
      </div>
    </div>
  );
};

export default Header;
